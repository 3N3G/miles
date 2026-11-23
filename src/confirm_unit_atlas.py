"""A fixed-identity, new-article test of native unit-to-reader predictions."""

import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from native_unit_atlas import sensitivity
from native_unit_local_replay import Capture,scores,edited_expert
from native_unit_routing import UnitControl,run


@torch.no_grad()
def edit_local(layer,reader,capture,visit,unit,random_seed=None):
    j=visit-1;args=capture.expert_args[j];units=[(unit['expert'],unit['unit'])];audit=[]
    if random_seed is None:y=edited_expert(layer.mlp.experts,args,units)
    else:
        control=UnitControl(units,visits=[1],random_seed=random_seed+visit-1)
        handles=[layer.mlp.experts.register_forward_pre_hook(control.begin),layer.mlp.experts.act_fn.register_forward_hook(control.activation),layer.mlp.experts.register_forward_hook(control.end)]
        try:y=layer.mlp.experts(*args)
        finally:
            for h in handles:h.remove()
        assert control.calls==1;audit=control.audit
    conv=layer.post_mlp_convolution(y.reshape_as(capture.pre[j]),**capture.conv_kwargs[j])
    block=(capture.pre[j].float()+capture.attn[j].float()+conv).to(capture.pre[j].dtype)
    _,alpha=scores(reader,capture.reader[j][0],block)
    return alpha[0,:,0].cpu().numpy(),audit


def main():
    from transformers import AutoModelForImageTextToText
    from pilot import MODEL,REVISION
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--units',type=Path,required=True)
    p.add_argument('--units-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert hashlib.sha256(a.units.read_bytes()).hexdigest()==a.units_sha256
    a.out.mkdir(parents=True,exist_ok=False);started=time.time();inputs=json.loads(a.inputs.read_text());spec=json.loads(a.units.read_text())
    units=spec['candidates']+spec['known_comparators'];assert len(units)==8
    assert sum(u['category']=='phase' for u in units)==4 and len(inputs['records'])==32
    (a.out/'frozen_units.json').write_bytes(a.units.read_bytes());(a.out/'frozen_inputs.json').write_bytes(a.inputs.read_bytes())
    model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'}).eval();model.requires_grad_(False)
    layer=model.model.language_model.layers[11];reader=model.model.language_model.layers[12].attention_input
    rows=[];checks=[]
    for index,record in enumerate(inputs['records']):
        ids=torch.tensor([record['token_ids']],device='cuda');capture=Capture(model)
        observed=UnitControl([(u['expert'],u['unit']) for u in units],visits=[],observe=True)
        try:
            with torch.no_grad():logits,baseline,_=run(model,ids,observed)
        finally:capture.close()
        (a.out/(record['id']+'-units.json')).write_text(json.dumps(observed.values)+'\n')
        if index==0:
            for label,control in [('identity',UnitControl([])),('absent',UnitControl([(128,0)]))]:
                with torch.no_grad():z,r,_=run(model,ids,control)
                assert torch.equal(z,logits) and np.array_equal(r,baseline)
                checks.append({'control':label,'maximum_logit_error':0,'maximum_reader_error':0});del z
        del logits
        predictions={visit:sensitivity(layer,reader,capture,visit) for visit in [1,3]}
        for unit in units:
            for visit in [1,3]:
                actual,_=edit_local(layer,reader,capture,visit,unit)
                if index==0:
                    with torch.no_grad():z,r,_=run(model,ids,UnitControl([(unit['expert'],unit['unit'])],visits=[visit]))
                    error=float(np.max(abs(r[visit-1]-actual)));assert error<=1e-6,error;del z
                    checks.append({'expert':unit['expert'],'unit':unit['unit'],'visit':visit,'maximum_full_forward_error':error})
                random=[];audits=[]
                if unit['category']=='phase' or (unit['expert'],unit['unit'])==(45,1127):
                    for s in [0,1,2]:
                        seed=1729+s*100000+index*10
                        r_local,audit=edit_local(layer,reader,capture,visit,unit,seed)
                        random.append(float((r_local[16:,-1]-baseline[visit-1,16:,-1]).mean()));audits.extend(audit)
                        if index==0 and s==0:
                            with torch.no_grad():z,r,_=run(model,ids,UnitControl([(unit['expert'],unit['unit'])],visits=[visit],random_seed=seed))
                            error=float(np.max(abs(r[visit-1]-r_local)));assert error<=1e-6,error;del z
                            checks.append({'control':'random','expert':unit['expert'],'unit':unit['unit'],'visit':visit,'maximum_full_forward_error':error})
                rows.append({'id':record['id'],'article':record['article'],**unit,'visit':visit,
                             'predicted_full':float(predictions[visit][0][unit['expert'],unit['unit']]),
                             'predicted_dot_only':float(predictions[visit][1][unit['expert'],unit['unit']]),
                             'actual_delta':float((actual[16:,-1]-baseline[visit-1,16:,-1]).mean()),
                             'random_deltas':random,'random_audits':audits,
                             'native_block2_weight':float(baseline[visit-1,16:,-1].mean())})
        (a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(a.out/'controls.json').write_text(json.dumps(checks,indent=2)+'\n')
        print(json.dumps({'event':'unit_atlas_confirmation_article','id':record['id'],'elapsed':time.time()-started}),flush=True)
    (a.out/'manifest.json').write_text(json.dumps({'status':'completed','wall_seconds':time.time()-started,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'model':MODEL,'revision':REVISION,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
         'units_sha256':a.units_sha256,'inputs_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':main()
