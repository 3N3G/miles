"""Development-only native expert causal screen and answer-format calibration."""

import argparse
import hashlib
import json
import subprocess
import time
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from inspect_native_experts import numeric_cue
from native_expert import RouterEdit, aliases, formatted_prompt, letter_metrics, render_question
from pilot import MODEL, REVISION, token_list


def write(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


@torch.no_grad()
def forward(model, ids, edit=None, full=False, zero=False):
    operator=model.model.language_model.layers[11].mlp
    handles=[]
    if edit is not None: handles.append(operator.gate.register_forward_hook(edit))
    if zero: handles.append(operator.register_forward_hook(lambda m,a,o:torch.zeros_like(o)))
    try:
        logits=model(input_ids=ids,use_cache=False,logits_to_keep=0 if full else 1).logits.float()
    finally:
        for h in handles:h.remove()
    if edit is not None: assert edit.calls==3,edit.calls
    return logits


def conditions(inputs):
    conditions=[('clean',None),('105-all',RouterEdit(105))]
    conditions += [(f'105-pass{p}',RouterEdit(105,visits=[p])) for p in [1,2,3]]
    conditions += [('105-renormalized',RouterEdit(105,renormalize=True))]
    conditions += [(f'{e}-all',RouterEdit(e)) for e in [45,3]+inputs['frequency_matched_experts']]
    return conditions


def save_routes(out,key,observer,ids):
    np.savez_compressed(out/(key+'.npz'), indices=np.stack([r[0] for r in observer.routes]),
                        weights=np.stack([r[1] for r in observer.routes]),token_ids=ids.cpu().numpy())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--rank',type=int,default=0);parser.add_argument('--world-size',type=int,default=1)
    parser.add_argument('--natural-text',type=Path);parser.add_argument('--gradients',action='store_true')
    args=parser.parse_args();root=args.out;root.mkdir(parents=True,exist_ok=True)
    args.out=root/f'rank{args.rank}';args.out.mkdir(exist_ok=False)
    started=time.time();inputs=json.loads(args.inputs.read_text());write(args.out/'frozen_inputs.json',inputs)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION);groups=aliases(tokenizer)
    if args.world_size>1:assert torch.cuda.device_count()==1,torch.cuda.device_count()
    write(args.out/'letter_aliases.json',{k:[{'id':i,'token':tokenizer.decode([i])} for i in v] for k,v in groups.items()})
    model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'}).eval()
    model.requires_grad_(False)
    calibration=[];controls=[]
    for record in inputs['records'] if args.rank==0 else []:
        if record['cohort']!='calibration':continue
        text,answer=render_question(record)
        for fmt in ['plain','chat_text']:
            rendered=formatted_prompt(tokenizer,text,fmt);ids=tokenizer(rendered,return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
            assert ids.shape[-1]<=512
            logits=forward(model,ids)[0,-1]
            if not controls:
                for name,edit in [('identity',RouterEdit()),('absent128',RouterEdit(128))]:
                    other=forward(model,ids,edit)[0,-1];assert torch.equal(logits,other),name
                    controls.append({'condition':name,'max_logit_error':float((logits-other).abs().max()),'visits':edit.calls})
                zero=forward(model,ids,zero=True)[0,-1]
                controls.append({'condition':'zero_entire_layer12_mlp','max_logit_change':float((zero-logits).abs().max()),**letter_metrics(zero,groups,answer)})
                write(args.out/'controls.json',controls)
            row={'id':record['id'],'subject':record['subject'],'format':fmt,'answer':answer,'rendered_input':rendered,'token_ids':ids[0].tolist(),
                 **letter_metrics(logits,groups,answer),'top_tokens':token_list(logits,tokenizer)}
            calibration.append(row)
            write(args.out/'calibration.json',calibration)
    if args.rank==0:
        scores={fmt:float(np.mean([r['gold_log_probability'] for r in calibration if r['format']==fmt])) for fmt in ['plain','chat_text']}
        fmt=max(scores,key=scores.get)
        write(args.out/'format_selection.json',{'format':fmt,'mean_gold_log_probability':scores,'source':'calibration only'})
        write(root/'format_selection.json',{'format':fmt,'mean_gold_log_probability':scores,'source':'calibration only'})
        print(json.dumps({'event':'format_selected','format':fmt,'scores':scores}),flush=True)
    else:
        while not (root/'format_selection.json').exists():
            assert time.time()-started < 300, 'Calibration worker did not finish'
            time.sleep(.5)
        fmt=json.loads((root/'format_selection.json').read_text())['format']
    rows=[]
    discovery=[r for r in inputs['records'] if r['cohort']=='discovery']
    for record in discovery[args.rank::args.world_size]:
        text,answer=render_question(record);rendered=formatted_prompt(tokenizer,text,fmt)
        ids=tokenizer(rendered,return_tensors='pt',add_special_tokens=False).input_ids.to('cuda');assert ids.shape[-1]<=512
        for name,edit in conditions(inputs):
            if name=='clean':edit=RouterEdit(observe=True)
            logits=forward(model,ids,edit)[0,-1]
            if name=='clean':
                save_routes(args.out,record['id'],edit,ids)
            rows.append({'id':record['id'],'subject':record['subject'],'condition':name,'answer':answer,
                         'rendered_input':rendered,'token_ids':ids[0].tolist(), 'audit':edit.audit if edit else [],
                         **letter_metrics(logits,groups,answer),'top_tokens':token_list(logits,tokenizer, k=5)})
        write(args.out/'question_results.json',rows)
        print(json.dumps({'event':'question','id':record['id'],'conditions':len(conditions(inputs)),'elapsed':time.time()-started}),flush=True)
    text_rows=[]
    texts=inputs['development_text']
    if args.natural_text:
        frozen_text=json.loads(args.natural_text.read_text());write(args.out/'frozen_natural_text.json',frozen_text)
        texts=texts+frozen_text['records']
    for record in texts[args.rank::args.world_size]:
        ids=torch.tensor([record['token_ids']],device='cuda')
        for name,edit in conditions(inputs):
            if name=='clean':edit=RouterEdit(observe=True)
            logits=forward(model,ids,edit,full=True)[0]
            logp=logits.log_softmax(-1);nll=-logp[:-1].gather(-1,ids[0,1:,None]).squeeze(-1)
            text_rows.append({'id':record['id'],'condition':name,'rendered_input':record['rendered_input'],
                              'token_ids':record['token_ids'],'cohort':record.get('cohort','development'),
                              'article':record.get('article'),'prediction_positions':list(range(16,len(record['token_ids'])-1)),
                              'nll':nll[16:].tolist(),'numeric_input':[numeric_cue(tokenizer.decode([i])) for i in record['token_ids'][16:-1]],
                              'numeric_target':[numeric_cue(tokenizer.decode([i])) for i in record['token_ids'][17:]],
                              'audit':edit.audit if edit else [],'top_tokens':token_list(logits[-1],tokenizer,k=5)})
            if name=='clean':save_routes(args.out,record['id'],edit,ids)
        write(args.out/'text_results.json',text_rows)
        print(json.dumps({'event':'text','id':record['id'],'elapsed':time.time()-started}),flush=True)
    if args.gradients:
        from native_expert_gradients import discover
        discover(model,tokenizer,groups,discovery[args.rank::args.world_size],fmt,args.out)
    write(args.out/'manifest.json',{'model':MODEL,'revision':REVISION,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                                   'inputs_sha256':hashlib.sha256(args.inputs.read_bytes()).hexdigest(),'wall_seconds':time.time()-started,
                                   'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),'status':'completed'})


if __name__=='__main__':main()
