"""Rank native units by their local, normalization-aware routing effects."""

import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from native_unit_local_replay import Capture,scores,edited_expert
from native_unit_routing import run


def sensitivity(layer,reader,capture,visit):
    j=visit-1;args=capture.expert_args[j];leaves=[]
    def leaf_hook(module,args,output):
        leaf=output.detach().requires_grad_(True);leaves.append(leaf);return leaf
    handle=layer.mlp.experts.act_fn.register_forward_hook(leaf_hook)
    try:y=layer.mlp.experts(*args)
    finally:handle.remove()
    assert len(leaves)==1 and torch.equal(y.detach(),capture.expert_out[j])
    residual=capture.pre[j].float()+capture.attn[j].float()
    conv=layer.post_mlp_convolution(y.reshape_as(capture.pre[j]),**capture.conv_kwargs[j])
    block=(residual+conv).to(capture.pre[j].dtype)
    assert torch.equal(block.detach(),capture.reader[j][1])
    base_scores,alpha=scores(reader,capture.reader[j][0],block)
    full=alpha[0,16:,0,-1].mean()
    d=block.shape[-1];rms=(block.float().square().mean(-1)+reader.norm.eps).sqrt().detach()
    numerator=(block.float()*reader.query.float().reshape(d)).sum(-1)*d**-.5
    last=(numerator/rms).unsqueeze(-1)
    only_num=torch.cat([base_scores[...,:-1].detach(),last.unsqueeze(-1)],-1).softmax(-1,dtype=torch.float32)
    grad,=torch.autograd.grad(full,leaves[0],retain_graph=True)
    grad_num,=torch.autograd.grad(only_num[0,16:,0,-1].mean(),leaves[0])
    groups=args[1].reshape(-1).sort().values
    result=[]
    for g in [grad,grad_num]:
        values=torch.zeros(128,1152,device=y.device,dtype=torch.float32)
        values.index_add_(0,groups,(-leaves[0].detach().float()*g.float()))
        result.append(values.cpu().numpy())
    assert np.isfinite(result).all()
    return result


@torch.no_grad()
def local_delta(layer,reader,capture,visit,units):
    j=visit-1;y=edited_expert(layer.mlp.experts,capture.expert_args[j],units)
    conv=layer.post_mlp_convolution(y.reshape_as(capture.pre[j]),**capture.conv_kwargs[j])
    block=(capture.pre[j].float()+capture.attn[j].float()+conv).to(capture.pre[j].dtype)
    _,alpha=scores(reader,capture.reader[j][0],block)
    _,base=scores(reader,*capture.reader[j])
    return float((alpha[0,16:,0,-1]-base[0,16:,0,-1]).mean())


def main():
    from transformers import AutoModelForImageTextToText
    from pilot import MODEL,REVISION
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);started=time.time();inputs=json.loads(a.inputs.read_text());records=inputs['records'];assert len(records)==16
    model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'}).eval();model.requires_grad_(False)
    layer=model.model.language_model.layers[11];reader=model.model.language_model.layers[12].attention_input
    all_scores=[]
    for index,record in enumerate(records):
        capture=Capture(model)
        try:
            with torch.no_grad():logits,_,_=run(model,torch.tensor([record['token_ids']],device='cuda'))
        finally:capture.close()
        all_scores.append(np.array([sensitivity(layer,reader,capture,v) for v in [1,2,3]]))
        if index==0:
            with torch.no_grad():replayed,_,_=run(model,torch.tensor([record['token_ids']],device='cuda'))
            assert torch.equal(logits,replayed),'Local gradients changed native model logits'
            assert all(p.grad is None for p in model.parameters())
            (a.out/'gradient_controls.json').write_text(json.dumps({'native_logits_unchanged_after_local_gradients':True,'pretrained_weight_gradients_absent':True})+'\n')
            del replayed
        del logits
        np.save(a.out/'unit_scores.npy',np.array(all_scores))
        print(json.dumps({'event':'unit_gradient_article','id':record['id'],'elapsed':time.time()-started}),flush=True)
    values=np.array(all_scores);mean=values.mean(0)[:,0];contrast=mean[2]-mean[0]
    phase=np.argwhere(mean[0]*mean[2]<0)
    phase=sorted(phase.tolist(),key=lambda x:(-abs(contrast[tuple(x)]),*x))
    stable=np.argwhere(mean[0]*mean[2]>0)
    stable=sorted(stable.tolist(),key=lambda x:(-np.mean(abs(mean[:,x[0],x[1]])),*x))
    candidates=[];seen={(45,1127),(3,664)}
    for category,ranked,n in [('phase',phase,4),('stable',stable,2)]:
        for expert,unit in ranked:
            if (expert,unit) in seen:continue
            candidates.append({'expert':expert,'unit':unit,'category':category,'predicted_per_pass':mean[:,expert,unit].tolist()});seen.add((expert,unit))
            if sum(r['category']==category for r in candidates)==n:break
    assert len(candidates)==6
    spec={'physical_layer':12,'selected_before_exact_validation':True,'candidates':candidates,
          'known_comparators':[{'expert':45,'unit':1127,'category':'known'},{'expert':3,'unit':664,'category':'known'}]}
    (a.out/'selected_units.json').write_text(json.dumps(spec,indent=2)+'\n')
    exact=[]
    for record in records:
        capture=Capture(model)
        try:
            with torch.no_grad():logits,_,_=run(model,torch.tensor([record['token_ids']],device='cuda'))
        finally:capture.close()
        del logits
        for unit in candidates+spec['known_comparators']:
            for visit in [1,3]:
                value=local_delta(layer,reader,capture,visit,[(unit['expert'],unit['unit'])])
                exact.append({'id':record['id'],**unit,'visit':visit,'exact_local_reader_delta':value})
        (a.out/'exact_validation.json').write_text(json.dumps(exact,indent=2)+'\n')
        print(json.dumps({'event':'unit_exact_article','id':record['id'],'elapsed':time.time()-started}),flush=True)
    (a.out/'manifest.json').write_text(json.dumps({'status':'completed','wall_seconds':time.time()-started,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'model':MODEL,'revision':REVISION,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'scope':'Development atlas on the existing 16 articles. No new independent claim.'},indent=2)+'\n')


if __name__=='__main__':main()
