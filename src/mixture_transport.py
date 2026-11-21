"""Simple pass-averaging controls for projected fixed-point readouts."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer
from pilot import MODEL,REVISION
from fit_transport import VISITS,prepare_items,score_predictions
from transport_math import centered,prediction_error


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);t=time.monotonic();torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    with safe_open(hf_hub_download(MODEL,'model-00001-of-00001.safetensors',revision=REVISION),framework='pt',device='cpu') as f:
        head=f.get_tensor('model.llm.unembed.weight')[:200058].to('cuda'); norm=f.get_tensor('model.llm.norm.weight').float().to('cuda')
    tokens=[tok.decode([i]) for i in range(len(head))];mask=torch.tensor([any(c.isalpha() or c.isdigit() for c in s) for s in tokens],device='cuda')
    items=prepare_items(json.loads((a.capture/'development_outputs.json').read_text()),tokens)
    dev=np.load(a.capture/'development_states.npz');native=torch.tensor(dev['logits'],device='cuda');hd=torch.tensor(dev['active'],device='cuda').float()
    cal=torch.from_numpy(np.load(a.capture/'calibration_active.npy')).to('cuda');target=torch.from_numpy(np.load(a.capture/'calibration_target.npy')).to('cuda');valid=torch.arange(128*48,144*48,device='cuda')
    rows=[];diagnostics=[]
    with torch.inference_mode():
        for seed in [0,1,2]:
            indices=np.random.default_rng(seed).choice(128,96,replace=False);flat=torch.tensor((indices[:,None]*48+np.arange(48)[None,:]).reshape(-1),device='cuda')
            my,sy,_=centered(target[flat]);stats={v:centered(cal[flat,v].float())[:2] for v in [11,17,23,29,35,41]}
            def std(x,v):m,s=stats[v];return (x-m)/s
            for visit in VISITS:
                for weight in [.5,.75]:
                    d=std(hd[:,visit],visit);v=std(cal[valid,visit].float(),visit)
                    if visit>=23:
                        d=weight*d+(1-weight)*std(hd[:,visit-12],visit-12);v=weight*v+(1-weight)*std(cal[valid,visit-12].float(),visit-12)
                    prediction=my+sy*d;validation=my+sy*v
                    meta={'mode':'active','method':f'cycle_mix_{weight}','seed':seed,'visit':visit}
                    rows.extend(score_predictions(prediction,native,items,tokens,mask,head,norm,meta));diagnostics.append({**meta,'validation_normalized_mse':prediction_error(validation,target[valid])})
    (a.out/'readouts.json').write_text(json.dumps(rows,indent=2)+'\n');(a.out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n');(a.out/'manifest.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'weights':[.5,.75],'elapsed_seconds':time.monotonic()-t},indent=2)+'\n')
if __name__=='__main__':main()
