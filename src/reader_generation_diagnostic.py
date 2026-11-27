"""Separate masks, generation preparation and native cache in a bounded check."""
import argparse,json,subprocess,time
from pathlib import Path
import torch
from complete_reader_capability import uncached


@torch.no_grad()
def cached(model,ids,count):
    result=[];out=model(input_ids=ids,use_cache=True,logits_to_keep=1)
    for i in range(count):
        idx=int(out.logits[0,-1].argmax());result.append(idx)
        if idx==200006 or i==count-1:break
        out=model(input_ids=torch.tensor([[idx]],device=ids.device),cache=out.cache,use_cache=True,logits_to_keep=1)
    return result


def main():
    from transformers import AutoModelForImageTextToText,AutoTokenizer
    from pilot import MODEL,REVISION
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4)
    inp=json.loads((a.reference/'frozen_inputs.json').read_text());prev={r['id']:r for r in json.loads((a.reference/'results.json').read_text())}
    records=[r for r in inp['records'] if r['family']=='arithmetic'][:2]
    (a.out/'frozen_inputs.json').write_bytes((a.reference/'frozen_inputs.json').read_bytes())
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'}).eval().requires_grad_(False)
    ids=torch.tensor([records[0]['token_ids']],device='cuda');checks={};outputs={}
    with torch.no_grad():
        base=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        for name,kw in [('ones_mask',{'attention_mask':torch.ones_like(ids),'use_cache':False}),('native_cache_prefill',{'use_cache':True})]:
            logits=model(input_ids=ids,logits_to_keep=1,**kw).logits[0,-1].float()
            checks[name]={'maximum_logit_error':float((logits-base).abs().max()),'same_top1':bool(logits.argmax()==base.argmax())}
        reference=uncached(model,ids,32);assert reference[:12]==prev[records[0]['id']]['generation']['token_ids']
        for name,use_cache in [('generate_uncached',False),('generate_cached',True)]:
            gen=model.generate(input_ids=ids,max_new_tokens=32,do_sample=False,use_cache=use_cache)[0,ids.shape[-1]:].cpu().tolist()
            outputs[name]={'token_ids':gen,'raw':tok.decode(gen),'exact32_match':gen==reference,
                           'first_mismatch':next((i for i,(x,y) in enumerate(zip(gen,reference)) if x!=y),None)}
        gen=cached(model,ids,32)
        outputs['manual_cached']={'token_ids':gen,'raw':tok.decode(gen),'exact32_match':gen==reference,
                                  'first_mismatch':next((i for i,(x,y) in enumerate(zip(gen,reference)) if x!=y),None)}
    outputs['manual_uncached']={'token_ids':reference,'raw':tok.decode(reference),'exact32_match':True}
    diagnosis={'prefill_checks':checks,'paths':outputs};(a.out/'diagnosis.json').write_text(json.dumps(diagnosis,indent=2)+'\n')
    print(json.dumps({'event':'generation_path_diagnosis',**diagnosis}),flush=True)
    rows=[]
    for r in records:
        ids=torch.tensor([r['token_ids']],device='cuda');gen=uncached(model,ids,256)
        assert gen[:12]==prev[r['id']]['generation']['token_ids']
        raw=tok.decode(gen);final=raw.rsplit('<|content_text|>',1)[-1] if '<|content_text|>' in raw else ''
        for marker in ['<|content_model_end_sampling|>','<|end_message|>']:final=final.split(marker)[0]
        rows.append({'id':r['id'],'prompt':r['model_input'],'target':r['target'],'intermediate':r['labels'],'token_ids':gen,
                     'raw':raw,'text':tok.decode(gen,skip_special_tokens=True),'final_text':final.strip(),'completed':gen[-1]==200006})
        (a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps({'event':'uncached_completion','id':r['id'],'raw':raw,'elapsed':time.time()-started}),flush=True)
    (a.out/'manifest.json').write_text(json.dumps({'status':'completed','model':MODEL,'revision':REVISION,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'wall_seconds':time.time()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated()},indent=2)+'\n')


if __name__=='__main__':main()
