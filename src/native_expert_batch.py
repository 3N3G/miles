"""Batched native expert confirmation, with calibration-only equivalence checks."""

import argparse
import hashlib
import json
import time
import subprocess
from pathlib import Path
import numpy as np
import torch
from native_expert import RouterEdit,aliases,formatted_prompt,letter_metrics,render_question
from native_expert_confirmation import OutputNormControl,make_masks,run,write


def encode_batch(tok,records,fmt,permutations):
    values=[]
    for record,permutation in zip(records,permutations):
        text,answer=render_question(record,permutation);rendered=formatted_prompt(tok,text,fmt)
        ids=tok(rendered,add_special_tokens=False).input_ids
        assert len(ids)<=512,(record['id'],len(ids))
        masks=make_masks(tok,rendered,record['question'],torch.tensor([ids]))
        values.append({'record':record,'rendered_input':rendered,'answer':answer,'ids':ids,'masks':masks,'permutation':permutation})
    longest=max(len(v['ids']) for v in values)
    # Right padding cannot enter an earlier valid causal prefix. Validate below.
    pad=tok.pad_token_id if tok.pad_token_id is not None else 0
    ids=torch.full((len(values),longest),pad,dtype=torch.long,device='cuda');attention=torch.zeros_like(ids)
    for i,v in enumerate(values):ids[i,:len(v['ids'])]=torch.tensor(v['ids'],device='cuda');attention[i,:len(v['ids'])]=1
    masks={}
    for key in values[0]['masks']:
        if key=='all':masks[key]=None;continue
        masks[key]=torch.zeros_like(ids,dtype=torch.bool)
        for i,v in enumerate(values):masks[key][i,:len(v['ids'])]=v['masks'][key].to('cuda')
    masks['all']=attention.bool()
    return values,ids,attention,masks


@torch.no_grad()
def batch_run(model,ids,attention,spec,masks,seed):
    handles=[];edits=[]
    try:
        for target in spec.get('targets',[]):
            operator=model.model.language_model.layers[target['layer_index']].mlp
            mask=masks.get(target.get('positions','all'));visits=target.get('visits',[1,2,3])
            if spec.get('kind') in ['matched_random','output_delete']:
                edit=OutputNormControl(target['expert'],visits,seed,mask,return_deleted=spec['kind']=='output_delete')
                handles.append(operator.experts.register_forward_hook(edit))
            else:
                edit=RouterEdit(target['expert'],visits=visits,renormalize=spec.get('renormalize',False),position_mask=mask)
                handles.append(operator.gate.register_forward_hook(edit))
            edits.append(edit)
        hidden=model.model(input_ids=ids,attention_mask=attention,use_cache=False).last_hidden_state
        # Match the official forward's BF16 head and FP32 softcap, with the same
        # one-row GEMM shape used for individual prompts.
        logits=[]
        for i,length in enumerate(attention.sum(-1).tolist()):
            h=hidden[i:i+1,length-1:length]
            logits.append(model.logit_softcap(model.lm_head(h).float())[0,0])
        logits=torch.stack(logits)
    finally:
        for handle in handles:handle.remove()
    assert all(edit.calls==3 for edit in edits)
    return logits,[audit for edit in edits for audit in edit.audit]


def validate(model,tok,groups,records,fmt,batch_size,spec,out):
    records=records[:4]+records[-4:];permutations=[list(range(4))]*len(records)
    values,ids,attention,masks=encode_batch(tok,records,fmt,permutations)
    before,_=batch_run(model,ids,attention,{},masks,1729)
    maximum=0.;max_gold=0.;same=[]
    first=next(c for c in spec['conditions'] if c.get('kind','router')=='router' and not c.get('renormalize'))
    batch_edited,_=batch_run(model,ids,attention,first,masks,1729)
    max_effect_error=0.
    for i,v in enumerate(values):
        single=torch.tensor([v['ids']],device='cuda');sm=v['masks']
        native,_=run(model,single,{},sm,1729)
        edited,_=run(model,single,first,sm,1729)
        m1=letter_metrics(before[i],groups,v['answer']);m2=letter_metrics(native,groups,v['answer'])
        same.append(m1['prediction']==m2['prediction'])
        maximum=max(maximum,float((before[i]-native).abs().max()))
        max_gold=max(max_gold,abs(m1['gold_choice_log_probability']-m2['gold_choice_log_probability']))
        a=letter_metrics(batch_edited[i],groups,v['answer'])['gold_choice_log_probability']-m1['gold_choice_log_probability']
        b=letter_metrics(edited,groups,v['answer'])['gold_choice_log_probability']-m2['gold_choice_log_probability']
        max_effect_error=max(max_effect_error,abs(a-b))
    identity,_=batch_run(model,ids,attention,{'targets':[{'layer_index':11,'expert':128}]},masks,1729)
    assert torch.equal(before,identity),'Batched identity changed logits'
    deletion,_=batch_run(model,ids,attention,{**first,'kind':'output_delete'},masks,1729)
    assert torch.equal(batch_edited,deletion),'Batched native deletion implementations differ'
    passed=all(same) and max_gold<=.05 and max_effect_error<=.05
    result={'calibration_examples':len(records),'same_predictions':sum(same),'max_logit_error':maximum,
            'max_choice_gold_log_probability_error':max_gold,'max_deletion_effect_error':max_effect_error,
            'identity_error':0,'router_vs_output_deletion_error':0,'tolerances_nats':.05,
            'batch_size_tested':len(records),'passed':passed,'fallback_if_failed':'single prompt forwards, no padding'}
    write(out/'batch_validation.json',result)
    return passed


def main():
    from transformers import AutoModelForImageTextToText,AutoTokenizer
    from pilot import MODEL,REVISION,token_list
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--cohort',default='confirmation');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);started=time.time()
    spec=json.loads(a.spec.read_text());inputs=json.loads(a.inputs.read_text());write(a.out/'frozen_spec.json',spec);write(a.out/'frozen_inputs.json',inputs)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION);groups=aliases(tok)
    model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'}).eval();model.requires_grad_(False)
    calibration=[r for r in inputs['records'] if r['cohort']=='calibration']
    batched=validate(model,tok,groups,calibration,spec['format'],a.batch_size,spec,a.out)
    assert batched,'Calibration equivalence failed. No confirmation question was run. Use a single-prompt stage.'
    batch_size=a.batch_size if batched else 1
    records=[r for r in inputs['records'] if r['cohort']==a.cohort];rows=[]
    for seed in a.seeds:
        rng=np.random.default_rng(seed);permutations=[rng.permutation(4).tolist() for _ in records]
        for start in range(0,len(records),batch_size):
            selected=records[start:start+batch_size]
            values,ids,attention,masks=encode_batch(tok,selected,spec['format'],permutations[start:start+batch_size])
            for condition in [{'name':'clean'}]+spec['conditions']:
                if batched:
                    logits,audit=batch_run(model,ids,attention,condition,masks,1729+seed*100000+start)
                else:
                    single=torch.tensor([values[0]['ids']],device='cuda')
                    scores,audit=run(model,single,condition,values[0]['masks'],1729+seed*100000+start);logits=scores[None]
                for i,v in enumerate(values):
                    row={'id':v['record']['id'],'subject':v['record']['subject'],'condition':condition['name'],'seed':seed,
                         'answer':v['answer'],'permutation':v['permutation'],'rendered_input':v['rendered_input'],'token_ids':v['ids'],
                         'batch_start':start,'batch_level_audit':audit,**letter_metrics(logits[i],groups,v['answer']),
                         'top_tokens':token_list(logits[i],tok,k=5)}
                    rows.append(row)
            write(a.out/'results.json',rows)
            print(json.dumps({'event':'question_batch','cohort':a.cohort,'seed':seed,'start':start,'size':len(values),'elapsed':time.time()-started}),flush=True)
    write(a.out/'manifest.json',{'status':'completed','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                               'model':MODEL,'revision':REVISION,'wall_seconds':time.time()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
                               'spec_sha256':hashlib.sha256(a.spec.read_bytes()).hexdigest(),'batch_size_used':batch_size,'seeds':a.seeds})


if __name__=='__main__':main()
