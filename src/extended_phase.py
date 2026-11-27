"""Test fixed native units over 24 passes, with native-prefix and replay guards."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from confirm_unit_atlas import edit_local
from native_unit_atlas import sensitivity
from native_unit_local_replay import Capture
from native_unit_routing import UnitControl,ReaderObserver


class ExtendedCapture(Capture):
    def close(self,count):
        for h in self.handles:h.remove()
        for name in ['pre','attn','expert_args','expert_out','conv','conv_kwargs','reader','norm_inputs']:
            assert len(getattr(self,name))==count,(name,len(getattr(self,name)),count)


@torch.no_grad()
def run(model,ids,count,control=None,retain=False):
    llm=model.model.language_model;old=llm.config.loop_num_passes
    capture=ExtendedCapture(model) if retain else None
    observer=ReaderObserver();bank3=[];final=[]
    handles=[llm.layers[12].attention_input.register_forward_hook(observer)]
    if retain:
        handles.append(llm.layers[17].register_forward_hook(lambda m,a,o:bank3.append(o.detach().clone())))
    handles.append(llm.final_input.register_forward_hook(lambda m,a,o:final.append(o[0].detach())))
    if control:
        operator=llm.layers[11].mlp.experts
        handles += [operator.register_forward_pre_hook(control.begin),operator.act_fn.register_forward_hook(control.activation),
                    operator.register_forward_hook(control.end)]
    try:
        llm.config.loop_num_passes=count
        logits=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0].float()
        rebuilt=model.logit_softcap(model.lm_head(llm.norm(final[0][:,-1:])).float())[0]
        assert torch.equal(rebuilt,logits),'Final head reconstruction differs'
    finally:
        llm.config.loop_num_passes=old
        for h in handles:h.remove()
        if capture:capture.close(count)
    assert len(observer.rows)==count
    if control:assert control.calls==count
    states=None
    if retain:
        assert len(bank3)==count
        states=torch.stack([torch.stack([capture.reader[j][1][0],bank3[j][0]]) for j in range(count)])
    return logits,np.stack(observer.rows),capture,states


def main():
    from transformers import AutoModelForImageTextToText
    from pilot import MODEL,REVISION
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--units',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4)
    inputs=json.loads(a.inputs.read_text());spec=json.loads(a.units.read_text());units=spec['candidates']+spec['known_comparators']
    assert len(units)==8 and len(inputs['records'])==16
    (a.out/'frozen_inputs.json').write_bytes(a.inputs.read_bytes());(a.out/'frozen_units.json').write_bytes(a.units.read_bytes())
    model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'}).eval()
    model.requires_grad_(False);layer=model.model.language_model.layers[11];reader=model.model.language_model.layers[12].attention_input
    rows=[];checks=[]
    for index,record in enumerate(inputs['records']):
        ids=torch.tensor([record['token_ids']],device='cuda')
        observe=UnitControl([(u['expert'],u['unit']) for u in units],visits=[],observe=True)
        logits,base,capture,states=run(model,ids,24,observe,retain=True)
        native,native_base,native_capture,native_states=run(model,ids,3,retain=True)
        assert torch.equal(states[:3],native_states) and np.array_equal(base[:3],native_base)
        for j in range(3):
            assert torch.equal(capture.expert_args[j][0],native_capture.expert_args[j][0])
            assert torch.equal(capture.expert_args[j][1],native_capture.expert_args[j][1])
        checks.append({'id':record['id'],'check':'native_three_pass_prefix','maximum_state_error':0,'maximum_reader_error':0})
        del native,native_capture,native_states
        # Store BF16 bit patterns losslessly, without converting through float16.
        np.savez_compressed(a.out/(record['id']+'-states.npz'),states_bf16_bits=states.view(torch.uint16).cpu().numpy(),
                            reader_weights=base,token_ids=np.array(record['token_ids']))
        (a.out/(record['id']+'-units.json')).write_text(json.dumps(observe.values)+'\n');del states
        if index==0:
            for stop in [1,2,6]:
                z,r,short,ss=run(model,ids,stop,retain=True)
                assert np.array_equal(base[:stop],r)
                for j in range(stop):assert torch.equal(short.reader[j][1],capture.reader[j][1])
                checks.append({'check':'stopping_count_prefix','stopping_count':stop,'maximum_reader_error':0,'maximum_block2_error':0})
                del z,r,short,ss
            for label,control in [('identity',UnitControl([])),('absent',UnitControl([(128,0)]))]:
                z,r,_,_=run(model,ids,24,control)
                assert torch.equal(z,logits) and np.array_equal(r,base)
                checks.append({'check':label,'maximum_logit_error':0,'maximum_reader_error':0});del z
        for visit in range(1,25):
            forecasts=sensitivity(layer,reader,capture,visit)
            for unit in units:
                actual,_=edit_local(layer,reader,capture,visit,unit)
                if index==0 and (unit['expert'],unit['unit']) in [(45,1127),(52,585)] and visit in [1,2,3,4,6,12,24]:
                    z,r,_,_=run(model,ids,24,UnitControl([(unit['expert'],unit['unit'])],visits=[visit]))
                    err=float(np.max(abs(actual-r[visit-1])));assert err<=1e-6,err
                    checks.append({'check':'local_vs_full_forward',**unit,'visit':visit,'maximum_reader_error':err});del z
                rows.append({'id':record['id'],'article':record['article'],**unit,'visit':visit,
                             'predicted_full':float(forecasts[0][unit['expert'],unit['unit']]),
                             'predicted_dot_only':float(forecasts[1][unit['expert'],unit['unit']]),
                             'actual_delta':float((actual[16:,-1]-base[visit-1,16:,-1]).mean()),
                             'native_block2_weight':float(base[visit-1,16:,-1].mean())})
        if index==0:
            z,r,_,_=run(model,ids,24)
            assert torch.equal(z,logits) and all(p.grad is None for p in model.parameters())
            checks.append({'check':'local_gradients_do_not_change_native_logits','maximum_logit_error':0,'weight_gradients_absent':True});del z
        del capture,logits
        (a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(a.out/'controls.json').write_text(json.dumps(checks,indent=2)+'\n')
        print(json.dumps({'event':'extended_phase_article','id':record['id'],'elapsed':time.time()-started}),flush=True)
    (a.out/'manifest.json').write_text(json.dumps({'status':'completed','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'model':MODEL,'revision':REVISION,'passes':24,'article_count':16,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'wall_seconds':time.time()-started,'inputs_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
        'units_sha256':hashlib.sha256(a.units.read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':main()
