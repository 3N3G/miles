"""Compare same-phase rotation continuation, innovations, and scalar acceleration."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer
from pilot import MODEL, REVISION
from fit_transport import VISITS, prepare_items, score_predictions
from transport_math import centered, fit_rotation, fit_ridge, transport, prediction_error


def fixed_point(x1, x2, x3):
    d1, d2 = x2 - x1, x3 - x2
    ratio = (d1 * d2).sum(-1, keepdim=True) / d1.square().sum(-1, keepdim=True).clamp_min(1e-12)
    r = ratio.clamp(-0.9, 0.9)
    return x3 + r / (1 - r) * d2, ratio.squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    t = time.monotonic()
    torch.set_num_threads(4)
    # A scalar contraction from 0 toward 1, with ratio 1/2, must extrapolate to 1.
    toy, _ = fixed_point(torch.zeros(1, 3), torch.full((1, 3), .5), torch.full((1, 3), .75))
    assert torch.allclose(toy, torch.ones_like(toy))
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    fn = hf_hub_download(MODEL, 'model-00001-of-00001.safetensors', revision=REVISION)
    with safe_open(fn, framework='pt', device='cpu') as f:
        head = f.get_tensor('model.llm.unembed.weight')[:200058].to('cuda')
        norm = f.get_tensor('model.llm.norm.weight').float().to('cuda')
    tokens = [tok.decode([i]) for i in range(len(head))]
    mask = torch.tensor([any(c.isalpha() or c.isdigit() for c in s) for s in tokens], device='cuda')
    items = prepare_items(json.loads((a.capture/'development_outputs.json').read_text()), tokens)
    dev = np.load(a.capture/'development_states.npz')
    native = torch.tensor(dev['logits'], device='cuda')
    cal = torch.from_numpy(np.load(a.capture/'calibration_active.npy')).to('cuda')
    target = torch.from_numpy(np.load(a.capture/'calibration_target.npy')).to('cuda')
    valid = torch.arange(128*48, 144*48, device='cuda')
    development = torch.tensor(dev['active'], device='cuda').float()
    rows, diagnostics = [], []
    with torch.inference_mode():
        for seed in [0,1,2]:
            para = np.random.default_rng(seed).choice(128,96,replace=False)
            index = torch.tensor((para[:,None]*48+np.arange(48)[None,:]).reshape(-1), device='cuda')
            my, sy, _ = centered(target[index])
            stats = {v:centered(cal[index,v].float())[:2] for v in [5,11,17,23,29,35,41]}
            def std(x,v):
                mu, scale = stats[v]
                return (x-mu)/scale
            def restore(x): return my+sy*x
            rotation = fit_rotation(cal[index,11].float(), cal[index,23].float())
            ridge_options=[]
            for alpha in [.01,.1,1]:
                fit=fit_ridge(cal[index,11].float(),cal[index,23].float(),alpha)
                prediction=std(cal[valid,23].float(),23)@fit['matrix']
                ridge_options.append((prediction_error(prediction,std(cal[valid,35].float(),35)),alpha,fit))
            _,alpha,ridge=min(ridge_options,key=lambda x:x[0])
            for visit in VISITS:
                xd=development[:,visit]
                xv=cal[valid,visit].float()
                predictions={'cycle_rotation':restore(std(xd,visit)), 'cycle_ridge':restore(std(xd,visit))}
                validation={k:restore(std(xv,visit)) for k in predictions}
                if visit in [11,23]:
                    remaining=(35-visit)//12
                    for label,fit in [('cycle_rotation',rotation),('cycle_ridge',ridge)]:
                        pdn=std(xd,visit)@torch.linalg.matrix_power(fit['matrix'],remaining)
                        pvn=std(xv,visit)@torch.linalg.matrix_power(fit['matrix'],remaining)
                        predictions[label]=restore(pdn)
                        validation[label]=restore(pvn)
                if visit>=23:
                    dd=std(xd,visit)-std(development[:,visit-12],visit-12)
                    dv=std(xv,visit)-std(cal[valid,visit-12].float(),visit-12)
                else:
                    dd,dv=torch.zeros_like(xd),torch.zeros_like(xv)
                predictions['cycle_change'],predictions['cycle_removed']=restore(dd),restore(-dd)
                validation['cycle_change'],validation['cycle_removed']=restore(dv),restore(-dv)
                ratio=None
                if visit>=35:
                    pd,ratio=fixed_point(std(development[:,visit-24],visit-24),std(development[:,visit-12],visit-12),std(xd,visit))
                    pv,_=fixed_point(std(cal[valid,visit-24].float(),visit-24),std(cal[valid,visit-12].float(),visit-12),std(xv,visit))
                    predictions['projected_fixed_point'],validation['projected_fixed_point']=restore(pd),restore(pv)
                else:
                    predictions['projected_fixed_point'],validation['projected_fixed_point']=restore(std(xd,visit)),restore(std(xv,visit))
                for method,pred in predictions.items():
                    meta={'mode':'active','method':method,'seed':seed,'visit':visit}
                    rows.extend(score_predictions(pred,native,items,tokens,mask,head,norm,meta))
                    cycle_error=prediction_error((validation[method]-my)/sy,std(cal[valid,35].float(),35)) if visit in [11,23] and method in ['cycle_rotation','cycle_ridge'] else None
                    diagnostics.append({**meta,'validation_normalized_mse':prediction_error(validation[method],target[valid]),'validation_pass_three_standardized_mse':cycle_error,'ridge':alpha if method=='cycle_ridge' else None,'development_ratios':ratio.cpu().tolist() if ratio is not None and method=='projected_fixed_point' else None})
                (a.out/'readouts.json').write_text(json.dumps(rows,indent=2)+'\n')
                (a.out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
                print(json.dumps({'seed':seed,'visit':visit,'elapsed':time.monotonic()-t}),flush=True)
    (a.out/'manifest.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'visits':VISITS,'seeds':[0,1,2],'elapsed_seconds':time.monotonic()-t},indent=2)+'\n')

if __name__=='__main__': main()
