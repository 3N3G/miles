"""Repair completion censoring with validated native cached greedy generation."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import torch


@torch.no_grad()
def uncached(model,ids,count):
    current=ids;result=[]
    for _ in range(count):
        logits=model(input_ids=current,use_cache=False,logits_to_keep=1).logits[0,-1]
        idx=int(logits.argmax());result.append(idx)
        if idx==200006:break
        current=torch.cat([current,torch.tensor([[idx]],device=ids.device)],-1)
    return result


def main():
    from transformers import AutoModelForImageTextToText,AutoTokenizer
    from pilot import MODEL,REVISION
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4)
    inputs=json.loads((a.reference/'frozen_inputs.json').read_text());previous={r['id']:r for r in json.loads((a.reference/'results.json').read_text())}
    records=[r for r in inputs['records'] if r['family'] in ['arithmetic','chain','practice']]
    if a.diagnostic:records=records[:2]
    (a.out/'frozen_inputs.json').write_bytes((a.reference/'frozen_inputs.json').read_bytes())
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'}).eval().requires_grad_(False)
    rows=[];checks=[]
    for index,r in enumerate(records):
        ids=torch.tensor([r['token_ids']],device='cuda');begin=time.time()
        with torch.no_grad():
            out=model.generate(input_ids=ids,max_new_tokens=512,do_sample=False,use_cache=True)
            generated=out[0,ids.shape[-1]:].cpu().tolist()
            prefix=previous[r['id']]['generation']['token_ids']
            assert generated[:len(prefix)]==prefix,('cached prefix mismatch',r['id'])
            checks.append({'id':r['id'],'check':'cached_matches_saved_uncached_prefix','tokens':len(prefix),'exact_match':True})
            if a.diagnostic and index==0:
                reference=uncached(model,ids,32)
                assert generated[:len(reference)]==reference,'Extended cached/uncached mismatch'
                checks.append({'id':r['id'],'check':'cached_matches_extended_uncached_prefix','tokens':len(reference),'exact_match':True})
        raw=tokenizer.decode(generated);final=raw.rsplit('<|content_text|>',1)[-1] if '<|content_text|>' in raw else ''
        for marker in ['<|content_model_end_sampling|>','<|end_message|>']:final=final.split(marker)[0]
        row={'id':r['id'],'family':r['family'],'token_ids':generated,'raw':raw,'text':tokenizer.decode(generated,skip_special_tokens=True),
             'final_text':final.strip(),'completed':bool(generated and generated[-1]==200006),'seconds':time.time()-begin}
        rows.append(row);(a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(a.out/'controls.json').write_text(json.dumps(checks,indent=2)+'\n')
        print(json.dumps({'event':'reader_capability_completion','id':r['id'],'generated_tokens':len(generated),'completed':row['completed'],
                          'final_text':row['final_text'],'seconds':row['seconds'],'elapsed':time.time()-started}),flush=True)
    (a.out/'manifest.json').write_text(json.dumps({'status':'completed','diagnostic':a.diagnostic,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'model':MODEL,'revision':REVISION,'max_new_tokens':512,'native_cache':True,'wall_seconds':time.time()-started,'questions':len(rows),
        'peak_gpu_bytes':torch.cuda.max_memory_allocated()},indent=2)+'\n')


if __name__=='__main__':main()
