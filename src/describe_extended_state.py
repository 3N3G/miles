"""Post-hoc controls for a global component behind apparent state recurrence."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_unit_atlas import estimate


def compare(state):
    tail=np.arange(18,24)
    return {str(lag):float(np.sqrt(np.mean((state[tail]-state[tail-lag])**2,axis=(1,2,3))).mean()) for lag in [1,3]}


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    inputs=json.loads((a.run/'frozen_inputs.json').read_text());rows=[]
    for r in inputs['records']:
        with np.load(a.run/(r['id']+'-states.npz')) as f:raw=(f['states_bf16_bits'].astype(np.uint32)<<16).view(np.float32)
        norm=raw/np.sqrt(np.mean(raw**2,axis=-1,keepdims=True)+1e-6)
        center=norm-norm.mean(axis=2,keepdims=True)
        no2027=raw.copy();no2027[...,2027]=0
        no2027/=np.sqrt(np.mean(no2027**2,axis=-1,keepdims=True)+1e-6)
        center_no2027=no2027-no2027.mean(axis=2,keepdims=True)
        rows.append({'id':r['id'],'article':r['article'],'original':compare(norm),'without_coordinate2027':compare(no2027),
                     'centered_across_tokens':compare(center),'without2027_then_centered':compare(center_no2027),
                     'late_normalized_energy_after_token_centering':float(np.mean(center[18:]**2)),
                     'late_raw_energy_share_coordinate2027':float(np.mean(raw[18:,...,2027]**2/np.sum(raw[18:]**2,axis=-1)))})
    draw=np.random.default_rng(1729).integers(0,16,(10000,16));summary={}
    for method in ['original','without_coordinate2027','centered_across_tokens','without2027_then_centered']:
        one=np.array([r[method]['1'] for r in rows]);three=np.array([r[method]['3'] for r in rows])
        summary[method]={'lag1':estimate(one,draw),'lag3':estimate(three,draw),'paired_difference':estimate(three-one,draw),
                         'mean_article_ratio_lag3_to_lag1':float((three/one).mean()),'articles_with_lag3_less_than_lag1':int((three<one).sum())}
    result={'status':'Exploratory controls after primary results. Descriptive 95% intervals, no new confirmatory family.',
            'purpose':'Check whether recurrence disappears when one known outlier or the within-pass component shared across tokens is removed.',
            'summary':summary,'per_article':rows,
            'mean_remaining_energy_after_token_centering':float(np.mean([r['late_normalized_energy_after_token_centering'] for r in rows])),
            'mean_energy_share_coordinate2027':float(np.mean([r['late_raw_energy_share_coordinate2027'] for r in rows])),
            'limits':['Centering removes the per-pass, per-block mean across128 token positions. It does not identify a semantic subspace.',
                      'Centered states are not renormalized, so lower absolute distances partly reflect removed energy.',
                      'Removing coordinate2027 leaves other possible control coordinates intact.',
                      'These transformations are offline diagnostics, not competition readouts.']}
    (a.run/'state_component_description.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['summary','mean_remaining_energy_after_token_centering','mean_energy_share_coordinate2027']},indent=2))


if __name__=='__main__':main()
