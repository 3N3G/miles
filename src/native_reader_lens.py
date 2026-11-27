"""Unfitted comparison of two single-visit native token read sites."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
import torch


def cleaned_top10(scores,tokenizer):
    special=set(tokenizer.all_special_ids);seen=set();result=[]
    for idx in torch.argsort(scores,descending=True).cpu().tolist():
        if idx in special:continue
        token=tokenizer.decode([idx]);key=token.strip().casefold()
        if not key or key in seen or len(token)>64:continue
        seen.add(key);result.append({'id':idx,'token':token})
        if len(result)==10:return result
    raise AssertionError('Insufficient vocabulary')


@torch.no_grad()
def trace(model,ids,visits,tokenizer):
    llm=model.model.language_model;counter=[-1];rows={'partial_buffer':[],'native_attention_reader':[]};final=[]
    def head(h):return model.lm_head(llm.norm(h).to(model.lm_head.weight.dtype))[0,-1].float()
    def enter(module,args):
        counter[0]+=1
        if counter[0] in visits:rows['partial_buffer'].append({'visit':counter[0],'tokens':cleaned_top10(head(args[0][:,-1:]),tokenizer)})
    def reader(module,args,output):
        if counter[0] in visits:rows['native_attention_reader'].append({'visit':counter[0],'tokens':cleaned_top10(head(output[0][:,-1:]),tokenizer)})
    handles=[]
    for layer in llm.layers:
        handles.extend([layer.register_forward_pre_hook(enter),layer.attention_input.register_forward_hook(reader)])
    handles.append(llm.final_input.register_forward_hook(lambda m,a,o:final.append(o[0][:,-1:].detach())))
    try:
        logits=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        rebuilt=model.logit_softcap(head(final[0]));assert torch.equal(rebuilt,logits),'Native head mismatch'
    finally:
        for h in handles:h.remove()
    assert counter[0]==47
    assert all([r['visit'] for r in rs]==visits for rs in rows.values())
    return logits,rows


@torch.no_grad()
def generate(model,ids,initial,tokenizer):
    generated=[];next_logits=initial;eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos]);current=ids
    for step in range(12):
        idx=int(next_logits.argmax());generated.append(idx)
        if idx in eos:break
        if step==11:break
        current=torch.cat([current,torch.tensor([[idx]],device=current.device)],-1)
        next_logits=model(input_ids=current,use_cache=False,logits_to_keep=1).logits[0,-1].float()
    return {'token_ids':generated,'text':tokenizer.decode(generated,skip_special_tokens=True),'raw':tokenizer.decode(generated)}


def main():
    from transformers import AutoModelForImageTextToText,AutoTokenizer
    from pilot import MODEL,REVISION
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);started=time.time()
    inputs=json.loads(a.inputs.read_text());visits=inputs['read_visits'];assert len(visits)==10
    (a.out/'frozen_inputs.json').write_bytes(a.inputs.read_bytes())
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'}).eval().requires_grad_(False)
    rows=[];controls=[]
    for index,record in enumerate(inputs['records']):
        ids=torch.tensor([record['token_ids']],device='cuda')
        assert tokenizer(record['model_input'],add_special_tokens=False).input_ids==record['token_ids']
        assert ids.shape[-1]-1==record['read_position']
        with torch.no_grad():
            logits,readouts=trace(model,ids,visits,tokenizer)
            if index==0:
                native=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()
                assert torch.equal(logits,native);controls.append({'check':'hooks_do_not_change_logits','maximum_error':0})
            output=generate(model,ids,logits,tokenizer)
        rows.append({'id':record['id'],'family':record['family'],'generation':output,'readouts':readouts})
        (a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(a.out/'controls.json').write_text(json.dumps(controls,indent=2)+'\n')
        print(json.dumps({'event':'native_reader_lens_question','id':record['id'],'family':record['family'],
                          'generation':output['text'],'elapsed':time.time()-started}),flush=True)
    (a.out/'manifest.json').write_text(json.dumps({'status':'completed','model':MODEL,'revision':REVISION,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'inputs_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
        'questions':len(rows),'wall_seconds':time.time()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated()},indent=2)+'\n')


if __name__=='__main__':main()
