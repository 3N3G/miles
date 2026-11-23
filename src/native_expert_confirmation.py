"""Frozen question confirmation with exact native edits and local norm controls."""

import argparse
import contextlib
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from native_expert import RouterEdit, aliases, formatted_prompt, letter_metrics, render_question
from inspect_native_experts import numeric_cue


def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)


class OutputNormControl:
    """Compare exact local expert deletion with equal-norm random output changes."""
    def __init__(self,expert,visits,seed,position_mask=None,return_deleted=False):
        self.expert=expert;self.visits=set(visits);self.seed=seed;self.position_mask=position_mask
        self.return_deleted=return_deleted;self.calls=0;self.audit=[]

    @torch.no_grad()
    def __call__(self,module,args,output):
        self.calls+=1
        if self.calls not in self.visits:return output
        x,indices,weights=args
        mask=indices==self.expert
        if self.position_mask is not None:mask=mask & self.position_mask.to(indices.device).reshape(-1,1)
        deleted=module.forward(x,indices,weights.masked_fill(mask,0))
        delta=deleted.float()-output.float();norm=delta.norm(dim=-1,keepdim=True)
        if self.return_deleted:edited=deleted
        else:
            generator=torch.Generator(device=output.device).manual_seed(self.seed+self.calls)
            random=torch.randn(output.shape,device=output.device,generator=generator)
            random=random/random.norm(dim=-1,keepdim=True).clamp_min(1e-20)
            edited=(output.float()+random*norm).to(output.dtype)
        actual=(edited.float()-output.float()).norm(dim=-1)
        self.audit.append({'visit':self.calls,'edited_tokens':int((norm.squeeze(-1)>0).sum()),
                           'reference_deletion_norm_sum':float(norm.sum()),'actual_edit_norm_sum':float(actual.sum()),
                           'norm_relative_error_total':float((actual.sum()-norm.sum()).abs()/norm.sum().clamp_min(1e-20))})
        return edited


@torch.no_grad()
def run(model,ids,spec,masks,seed):
    handles=[];edits=[]
    try:
        for target in spec.get('targets',[]):
            operator=model.model.language_model.layers[target['layer_index']].mlp
            mask=masks.get(target.get('positions','all'))
            visits=target.get('visits',[1,2,3])
            if spec.get('kind') in ['matched_random','output_delete']:
                edit=OutputNormControl(target['expert'],visits,seed,mask,return_deleted=spec['kind']=='output_delete')
                handles.append(operator.experts.register_forward_hook(edit))
            else:
                edit=RouterEdit(target['expert'],visits=visits,renormalize=spec.get('renormalize',False),position_mask=mask)
                handles.append(operator.gate.register_forward_hook(edit))
            edits.append(edit)
        logits=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()
    finally:
        for handle in handles:handle.remove()
    for edit in edits:assert edit.calls==3,edit.calls
    return logits,[audit for edit in edits for audit in edit.audit]


def make_masks(tokenizer,rendered,question,ids):
    encoded=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    assert encoded['input_ids']==ids[0].tolist()
    start=rendered.index(question);end=start+len(question)
    offsets=encoded['offset_mapping']
    q=torch.tensor([b>start and a<end for a,b in offsets])
    number=torch.tensor([numeric_cue(tokenizer.decode([int(t)])) for t in ids[0]])
    last=torch.zeros(len(offsets),dtype=torch.bool);last[-1]=True
    # Choices begin after the actual question and end before the assistant prefix.
    choice_start=rendered.index('\nA.',end)
    choice_end=rendered.index('\nAnswer:',choice_start)
    options=torch.tensor([b>choice_start and a<choice_end for a,b in offsets])
    return {'all':None,'question':q,'numeric':number,'question_numeric':q&number,
            'options':options,'last':last,'nonnumeric':~number}


def main():
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from pilot import MODEL, REVISION, token_list
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--cohort',default='confirmation');p.add_argument('--take',type=int)
    p.add_argument('--native-order',action='store_true');p.add_argument('--capture-neurons',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);started=time.time()
    spec=json.loads(a.spec.read_text());inputs=json.loads(a.inputs.read_text())
    write(a.out/'frozen_spec.json',spec);write(a.out/'frozen_inputs.json',inputs)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION);groups=aliases(tok)
    model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'}).eval()
    model.requires_grad_(False)
    records=[r for r in inputs['records'] if r['cohort']==a.cohort]
    if a.take is not None:records=records[:a.take]
    rows=[];controls=[];rng=np.random.default_rng(a.seed)
    for index,record in enumerate(records):
        permutation=list(range(4)) if a.native_order else rng.permutation(4).tolist()
        text,answer=render_question(record,permutation)
        rendered=formatted_prompt(tok,text,spec['format']);ids=tok(rendered,return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
        assert ids.shape[-1]<=512,(record['id'],ids.shape[-1])
        masks=make_masks(tok,rendered,record['question'],ids)
        baseline,_=run(model,ids,{},masks,1729)
        if index==0:
            sham={'targets':[{'layer_index':11,'expert':128}]}
            identity,_=run(model,ids,sham,masks,1729);assert torch.equal(baseline,identity)
            first=next(s for s in spec['conditions'] if s.get('targets') and s.get('kind','router')=='router' and not s.get('renormalize'))
            deleted,_=run(model,ids,first,masks,1729)
            equivalent={**first,'kind':'output_delete'}
            output_deleted,_=run(model,ids,equivalent,masks,1729)
            assert torch.equal(deleted,output_deleted),'Output replay differs from exact routing-weight removal'
            controls.append({'identity_logit_error':0,'router_vs_output_deletion_logit_error':0})
            write(a.out/'controls.json',controls)
        for cond in [{'name':'clean'}]+spec['conditions']:
            random_seed=1729+a.seed*100000+int(hashlib.sha256(record['id'].encode()).hexdigest()[:8],16)%100000
            logits,audit=(baseline,[]) if cond['name']=='clean' else run(model,ids,cond,masks,random_seed)
            row={'id':record['id'],'subject':record['subject'],'cohort':a.cohort,'seed':a.seed,'condition':cond['name'],
                 'permutation':permutation,'answer':answer,'rendered_input':rendered,'token_ids':ids[0].tolist(),
                 'audit':audit,**letter_metrics(logits,groups,answer),'top_tokens':token_list(logits,tok,k=5)}
            rows.append(row)
        write(a.out/'results.json',rows)
        print(json.dumps({'event':'confirmed_question','id':record['id'],'index':index,'seed':a.seed,'elapsed':time.time()-started}),flush=True)
    if a.capture_neurons:
        from capture_native_neurons import capture
        source=json.loads(a.capture_neurons.read_text())
        capture(model,[r for r in source['records'] if r['split'] in ['fit','validation']],a.out/'native_neurons',experts=(105,45,3,89,41,90))
    write(a.out/'manifest.json',{'model':MODEL,'revision':REVISION,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'spec_sha256':hashlib.sha256(a.spec.read_bytes()).hexdigest(),'inputs_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
         'wall_seconds':time.time()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'status':'completed'})


if __name__=='__main__':main()
