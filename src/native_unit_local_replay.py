"""Reconstruct exact unit-to-reader interventions and decompose the reader's RMS."""

import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from native_unit_routing import UnitControl,run


class Capture:
    def __init__(self,model):
        self.pre=[];self.attn=[];self.expert_args=[];self.expert_out=[];self.conv=[];self.conv_kwargs=[];self.reader=[]
        layer=model.model.language_model.layers[11];self.handles=[]
        self.handles.append(layer.register_forward_pre_hook(lambda m,a:self.pre.append(a[0].detach().clone())))
        self.handles.append(layer.post_attention_convolution.register_forward_hook(lambda m,a,o:self.attn.append(o.detach().clone())))
        self.handles.append(layer.mlp.experts.register_forward_pre_hook(lambda m,a:self.expert_args.append(tuple(x.detach().clone() for x in a))))
        self.handles.append(layer.mlp.experts.register_forward_hook(lambda m,a,o:self.expert_out.append(o.detach().clone())))
        def conv_pre(m,args,kwargs):
            assert kwargs.get('cache') is None
            self.conv_kwargs.append({k:v.detach().clone() if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()})
        self.handles.append(layer.post_mlp_convolution.register_forward_pre_hook(conv_pre,with_kwargs=True))
        self.handles.append(layer.post_mlp_convolution.register_forward_hook(lambda m,a,o:self.conv.append(o.detach().clone())))
        self.handles.append(model.model.language_model.layers[12].attention_input.register_forward_pre_hook(lambda m,a:self.reader.append(tuple(x.detach().clone() for x in a))))

    def close(self):
        for h in self.handles:h.remove()
        for name in ['pre','attn','expert_args','expert_out','conv','conv_kwargs','reader']:assert len(getattr(self,name))==3,name


def scores(module,blocks,partial):
    raw=torch.cat([blocks,partial.unsqueeze(-2)],-2).float();keys=module.norm(raw);d=keys.shape[-1]
    q=module.query.float().view(-1,d).expand(*keys.shape[:-2],-1,-1)
    score=torch.matmul(q,keys.transpose(2,3))*d**-.5
    return score,score.softmax(-1,dtype=torch.float32)


def edited_expert(module,args,units):
    control=UnitControl(units,visits=[1],direct=True)
    handles=[module.register_forward_pre_hook(control.begin),module.act_fn.register_forward_hook(control.activation),module.register_forward_hook(control.end)]
    try:output=module(*args)
    finally:
        for h in handles:h.remove()
    assert control.calls==1
    return output


def counterfactual_alpha(base_scores,last):
    return torch.cat([base_scores[...,:-1],last.unsqueeze(-1)],-1).softmax(-1,dtype=torch.float32)[...,-1]


@torch.no_grad()
def main():
    from transformers import AutoModelForImageTextToText
    from pilot import MODEL,REVISION
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);started=time.time();inputs=json.loads(a.inputs.read_text())
    model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'}).eval();model.requires_grad_(False)
    layer=model.model.language_model.layers[11];reader=model.model.language_model.layers[12].attention_input
    assert not layer.is_block_boundary
    rows=[];checks=[]
    for record in inputs['records']:
        ids=torch.tensor([record['token_ids']],device='cuda');capture=Capture(model)
        try:logits,observed,errors=run(model,ids)
        finally:capture.close()
        del logits
        saved=np.load(a.reference/(record['id']+'-native.npz'))['reader_weights']
        baseline_error=float(np.max(np.abs(saved-observed)));assert baseline_error<=1e-6,baseline_error
        for visit in [1,3]:
            v=visit-1;args=capture.expert_args[v];y=layer.mlp.experts(*args)
            assert torch.equal(y,capture.expert_out[v]),'Expert replay changed values'
            conv=layer.post_mlp_convolution(y.reshape_as(capture.pre[v]),**capture.conv_kwargs[v])
            assert torch.equal(conv,capture.conv[v]),'Convolution replay changed values'
            residual=capture.pre[v].float()+capture.attn[v].float()
            block=(residual+conv).to(capture.pre[v].dtype);blocks,partial=capture.reader[v]
            assert torch.equal(block,partial),'Residual reconstruction differs'
            s0,alpha0=scores(reader,blocks,block)
            assert float(np.max(abs(alpha0[0,:,0].cpu().numpy()-observed[v])))<=1e-6
            rms0=(block.float().square().mean(-1)+reader.norm.eps).sqrt()
            for name,units in [('unit45',[(45,1127)]),('unit3',[(3,664)]),('joint',[(45,1127),(3,664)]),('control45',[(45,130)])]:
                yp=edited_expert(layer.mlp.experts,args,units)
                cp=layer.post_mlp_convolution(yp.reshape_as(capture.pre[v]),**capture.conv_kwargs[v])
                bp=(residual+cp).to(capture.pre[v].dtype);s1,alpha1=scores(reader,blocks,bp)
                reference=np.load(a.reference/(record['id']+f'-{name}-pass{visit}.npz'))['reader_weights'][v]
                error=float(np.max(abs(reference-alpha1[0,:,0].cpu().numpy())));assert error<=1e-6,(record['id'],name,visit,error)
                rms1=(bp.float().square().mean(-1)+reader.norm.eps).sqrt()
                base=alpha0[...,-1];full=alpha1[...,-1]
                numerator_only=counterfactual_alpha(s0,s1[...,-1]*(rms1/rms0).unsqueeze(-1))
                rms_only=counterfactual_alpha(s0,s0[...,-1]*(rms0/rms1).unsqueeze(-1))
                numerator=.5*((numerator_only-base)+(full-rms_only))
                norm=.5*((rms_only-base)+(full-numerator_only))
                total=full-base
                sum_error=float((numerator+norm-total).abs().max());assert sum_error<=1e-7,sum_error
                row={'id':record['id'],'article':record['article'],'condition':name,'visit':visit,
                     'maximum_full_model_prediction_error':error,'maximum_decomposition_sum_error':sum_error,
                     'delta_block2_weight':total[0,16:,0].cpu().tolist(),
                     'numerator_contribution':numerator[0,16:,0].cpu().tolist(),'rms_contribution':norm[0,16:,0].cpu().tolist(),
                     'native_rms_mean':float(rms0[:,16:].mean()),'edited_rms_mean':float(rms1[:,16:].mean())}
                rows.append(row)
            checks.append({'id':record['id'],'visit':visit,'expert_replay_error':0,'convolution_replay_error':0,'block_replay_error':0,'baseline_saved_reader_error':baseline_error})
        (a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(a.out/'controls.json').write_text(json.dumps(checks,indent=2)+'\n')
        print(json.dumps({'event':'local_replay_article','id':record['id'],'elapsed':time.time()-started}),flush=True)
    (a.out/'manifest.json').write_text(json.dumps({'status':'completed','model':MODEL,'revision':REVISION,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'wall_seconds':time.time()-started,
        'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'scope':'Exploratory explanation of existing interventions, not a new independent test.'},indent=2)+'\n')


if __name__=='__main__':main()
