"""Observe native expert activations without replacing the native forward code."""

import json
from pathlib import Path
import numpy as np
import torch


class NativeNeuronObserver:
    def __init__(self,experts):
        self.experts=set(experts);self.visit=0;self.rows=[]

    def begin(self,module,args):
        _,self.indices,self.weights=args
        self.order=sorted(e for e in self.indices.unique().tolist() if e<128)
        self.group_ids,self.group_perm=torch.sort(self.indices.reshape(-1))
        self.group_tokens=self.group_perm//self.indices.shape[-1]
        self.group_weights=self.weights.reshape(-1)[self.group_perm]
        self.index=0;self.visit+=1

    def activation(self,module,args,output):
        if output.shape[0]==self.indices.numel():
            # The pinned runtime defaults to grouped_mm: one activation call
            # contains every selected token/expert pair, sorted by expert ID.
            assert self.index==0
            for expert in sorted(self.experts):
                mask=self.group_ids==expert
                if mask.any():
                    self.rows.append({'expert':expert,'visit':self.visit,
                        'positions':self.group_tokens[mask].detach().cpu().numpy(),
                        'weights':self.group_weights[mask].detach().cpu().numpy(),
                        'activation':output[mask].detach().float().cpu().numpy()})
            self.index=len(self.order)
            return
        expert=self.order[self.index];self.index+=1
        token,slot=torch.where(self.indices==expert)
        assert output.shape[0]==len(token),(expert,output.shape,len(token))
        if expert in self.experts:
            self.rows.append({'expert':expert,'visit':self.visit,'positions':token.detach().cpu().numpy(),
                              'weights':self.weights[token,slot].detach().cpu().numpy(),
                              'activation':output.detach().float().cpu().numpy()})

    def end(self,module,args,output):
        assert self.index==len(self.order),(self.index,len(self.order))


@torch.no_grad()
def capture(model,records,out,experts=(105,45,3)):
    out=Path(out);out.mkdir(exist_ok=False)
    operator=model.model.language_model.layers[11].mlp
    manifest=[]
    for index,record in enumerate(records):
        observer=NativeNeuronObserver(experts)
        ids=torch.tensor([record['token_ids']],device='cuda')
        baseline=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits if index==0 else None
        routes=[];route_calls={}
        def route_hook(layer):
            def observe(module,args,output):
                route_calls[layer]=route_calls.get(layer,0)+1
                routes.append((layer,route_calls[layer],output[2].detach().cpu().numpy(),output[1].detach().cpu().numpy()))
            return observe
        handles=[operator.experts.register_forward_pre_hook(observer.begin),
                 operator.experts.act_fn.register_forward_hook(observer.activation),
                 operator.experts.register_forward_hook(observer.end)]
        config=model.model.language_model.config
        for layer in range(config.loop_layer_index_start,config.loop_layer_index_end):
            handles.append(model.model.language_model.layers[layer].mlp.gate.register_forward_hook(route_hook(layer)))
        try:
            logits=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits
        finally:
            for handle in handles:handle.remove()
        assert observer.visit==3
        if baseline is not None:
            assert torch.equal(baseline,logits),'Observation changed the native logits'
        arrays={}
        for j,row in enumerate(observer.rows):
            for key,value in row.items():arrays[f'{j}_{key}']=value
        np.savez_compressed(out/f'{index:03}.npz',**arrays)
        np.savez_compressed(out/f'core_routing_{index:03}.npz',layer_indices=np.array([r[0] for r in routes]),
                            visits=np.array([r[1] for r in routes]),indices=np.stack([r[2] for r in routes]),weights=np.stack([r[3] for r in routes]))
        assert set(route_calls.values())=={config.loop_num_passes}
        manifest.append({'id':record['id'],'split':record.get('split',record.get('cohort')),
                         'rendered_input':record['rendered_input'],'token_ids':record['token_ids'],
                         'file':f'{index:03}.npz','expert_visit_rows':len(observer.rows)})
        (out/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({'event':'native_neuron_capture','id':record['id'],'index':index}),flush=True)
    norms={str(e):operator.experts.down_proj[e].float().norm(dim=0).cpu().numpy() for e in experts}
    np.savez_compressed(out/'decoder_column_norms.npz',**norms)
    (out/'controls.json').write_text(json.dumps({'native_observation_logit_error':0,'visits':3,
        'layer_index':11,'experts':list(experts),'method':'Unchanged native forward, parent and activation observation hooks'},indent=2)+'\n')


def main():
    import argparse
    import subprocess
    import time
    from transformers import AutoModelForImageTextToText
    from pilot import MODEL,REVISION
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);started=time.time()
    source=json.loads(a.inputs.read_text());records=[r for r in source['records'] if r['split'] in ['fit','validation']]
    assert len(records)==144
    model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'}).eval();model.requires_grad_(False)
    capture(model,records,a.out/'native_neurons',experts=(105,45,3,89,41,90))
    (a.out/'manifest.json').write_text(json.dumps({'status':'completed','model':MODEL,'revision':REVISION,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'wall_seconds':time.time()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated()},indent=2)+'\n')


if __name__=='__main__':main()
