"""Native grouped-unit edits and exact observation of a downstream block reader."""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
import numpy as np
import torch


class UnitControl:
    def __init__(self,units,visits=(1,2,3),random_seed=None,direct=False):
        self.units=units;self.visits=set(visits);self.random_seed=random_seed;self.direct=direct
        self.calls=0;self.inner=False;self.audit=[]

    def begin(self,module,args):
        _,indices,weights=args
        self.group_ids,self.perm=indices.reshape(-1).sort()
        self.calls+=1

    def activation(self,module,args,output):
        if self.calls not in self.visits or not (self.inner or self.direct):return
        assert len(output)==len(self.group_ids),'Expected the pinned grouped expert backend'
        changed=output.clone()
        for expert,unit in self.units:
            mask=self.group_ids==expert
            if mask.any():changed[mask,unit]=0
        return changed

    @torch.no_grad()
    def end(self,module,args,output):
        if self.direct or self.calls not in self.visits or not any((self.group_ids==e).any() for e,u in self.units):return
        self.inner=True
        try:deleted=module.forward(*args)
        finally:self.inner=False
        delta=deleted.float()-output.float();norm=delta.norm(dim=-1,keepdim=True)
        if self.random_seed is None:edited=deleted
        else:
            g=torch.Generator(device=output.device).manual_seed(self.random_seed+self.calls)
            random=torch.randn(output.shape,device=output.device,generator=g)
            random=random/random.norm(dim=-1,keepdim=True).clamp_min(1e-20)
            edited=(output.float()+random*norm).to(output.dtype)
        actual=(edited.float()-output.float()).norm(dim=-1)
        self.audit.append({'visit':self.calls,'intended_norm_sum':float(norm.sum()),'actual_norm_sum':float(actual.sum()),
                           'nonzero_token_edits':int((norm>0).sum()),
                           'relative_norm_sum_error':float((actual.sum()-norm.sum()).abs()/norm.sum().clamp_min(1e-20))})
        return edited


class ReaderObserver:
    def __init__(self):self.rows=[];self.errors=[]

    def __call__(self,module,args,output):
        blocks,partial=args
        keys=module.norm(torch.cat([blocks,partial.unsqueeze(-2)],dim=-2).float())
        d=keys.shape[-1];q=module.query.float().view(-1,d).expand(*keys.shape[:-2],-1,-1)
        weights=(torch.matmul(q,keys.transpose(2,3))*d**-.5).softmax(-1,dtype=torch.float32)
        reconstructed=torch.matmul(weights,keys)
        native=torch.stack(output,dim=-2)
        error=float((native-reconstructed).detach().abs().max())
        assert error<=1e-5,error
        assert weights.shape[-2]==1,weights.shape
        self.rows.append(weights[0,:,0].detach().cpu().numpy());self.errors.append(error)


@torch.no_grad()
def run(model,ids,control=None):
    observer=ReaderObserver();operator=model.model.language_model.layers[11].mlp.experts
    handles=[model.model.language_model.layers[12].attention_input.register_forward_hook(observer)]
    if control:
        handles += [operator.register_forward_pre_hook(control.begin),
                    operator.act_fn.register_forward_hook(control.activation),operator.register_forward_hook(control.end)]
    try:logits=model(input_ids=ids,use_cache=False).logits[0].float()
    finally:
        for h in handles:h.remove()
    assert len(observer.rows)==3
    if control:assert control.calls==3
    return logits,np.stack(observer.rows),observer.errors


def conditions():
    result=[]
    for visit in [1,3]:
        result.append((f'unit45-pass{visit}',[(45,1127)],[visit],None))
        result += [(f'unit45-pass{visit}-random{s}',[(45,1127)],[visit],s) for s in [0,1,2]]
    for label,units in [('unit3',[(3,664)]),('joint',[(45,1127),(3,664)]),('control45',[(45,130)])]:
        for visit in [1,3]:
            result += [(f'{label}-pass{visit}',units,[visit],None),
                       (f'{label}-pass{visit}-random0',units,[visit],0)]
    result += [('time-units-all',[(105,156),(105,811)],[1,2,3],None),
               ('time-units-all-random0',[(105,156),(105,811)],[1,2,3],0)]
    return result


def main():
    from transformers import AutoModelForImageTextToText
    from pilot import MODEL,REVISION
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);started=time.time();inputs=json.loads(a.inputs.read_text())
    (a.out/'frozen_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'}).eval();model.requires_grad_(False)
    rows=[];controls=[]
    for doc,record in enumerate(inputs['records']):
        ids=torch.tensor([record['token_ids']],device='cuda')
        baseline,reader,errors=run(model,ids);logp=baseline.log_softmax(-1);prob=logp.exp()
        if doc==0:
            untouched=model(input_ids=ids,use_cache=False).logits[0].float();assert torch.equal(untouched,baseline)
            for label,c in [('identity',UnitControl([])),('absent',UnitControl([(128,0)]))]:
                logits,r,_=run(model,ids,c);assert torch.equal(logits,baseline);assert np.array_equal(r,reader)
                controls.append({'condition':label,'max_logit_error':0,'max_reader_error':0})
            one,r,_=run(model,ids,UnitControl([(45,1127)],visits=[1]));direct,r2,_=run(model,ids,UnitControl([(45,1127)],visits=[1],direct=True))
            assert torch.equal(one,direct);assert np.array_equal(r,r2)
            controls.append({'condition':'native_direct_unit_vs_output_counterfactual','max_logit_error':0,'max_reader_error':0})
            (a.out/'controls.json').write_text(json.dumps(controls,indent=2)+'\n')
        np.savez_compressed(a.out/(record['id']+'-native.npz'),reader_weights=reader)
        for label,units,visits,seed in conditions():
            noise=None if seed is None else 1729+seed*100000+doc*10
            edit=UnitControl(units,visits=visits,random_seed=noise)
            logits,r,err=run(model,ids,edit)
            if visits==[3]:assert np.array_equal(r[:2],reader[:2]),'A late unit edit changed earlier reader weights'
            lp=logits.log_softmax(-1);kl=(prob*(logp-lp)).sum(-1)
            labels=ids[0,1:,None];base_nll=-logp[:-1].gather(-1,labels).squeeze(-1);nll=-lp[:-1].gather(-1,labels).squeeze(-1)
            rows.append({'id':record['id'],'article':record['article'],'condition':label,'units':units,'visits':visits,
                         'rendered_input':record['rendered_input'],'token_ids':record['token_ids'],
                         'reader_last_key_mean':[float(v[16:,-1].mean()) for v in r],
                         'native_reader_last_key_mean':[float(v[16:,-1].mean()) for v in reader],
                         'reader_reconstruction_max_error':max(err),'audit':edit.audit,
                         'nll':nll[16:].tolist(),'native_nll':base_nll[16:].tolist(),'kl':kl[16:].tolist(),
                         'native_top1_agreement':float((logits[16:].argmax(-1)==baseline[16:].argmax(-1)).float().mean())})
            np.savez_compressed(a.out/(record['id']+'-'+label+'.npz'),reader_weights=r)
        temp=a.out/'results.tmp';temp.write_text(json.dumps(rows,indent=2)+'\n');temp.replace(a.out/'results.json')
        print(json.dumps({'event':'unit_routing_article','id':record['id'],'index':doc,'elapsed':time.time()-started}),flush=True)
    (a.out/'manifest.json').write_text(json.dumps({'status':'completed','model':MODEL,'revision':REVISION,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'wall_seconds':time.time()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated()},indent=2)+'\n')


if __name__=='__main__':main()
