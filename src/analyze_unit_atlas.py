"""Fixed article-level analysis of native routing forecasts and phase replication."""

import argparse,json
from pathlib import Path
import numpy as np


def estimate(values,draw,family=1):
    values=np.asarray(values)
    return {'mean':float(values.mean()),'interval':np.quantile(values[draw].mean(-1),[.025/family,1-.025/family]).tolist(),
            'family_size':family,'sampling_unit':'article','articles':len(values),'resamples':10000,'seed':1729}


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    assert json.loads((a.run/'manifest.json').read_text())['status']=='completed'
    rows=json.loads((a.run/'results.json').read_text());spec=json.loads((a.run/'frozen_units.json').read_text())
    units=spec['candidates']+spec['known_comparators'];ids=sorted({r['id'] for r in rows});assert len(ids)==32 and len(rows)==32*16
    grouped={q:sorted([r for r in rows if r['id']==q],key=lambda r:(r['expert'],r['unit'],r['visit'])) for q in ids}
    for q,records in grouped.items():assert len({(r['expert'],r['unit'],r['visit']) for r in records})==16
    actual=np.array([[r['actual_delta'] for r in grouped[q]] for q in ids]);full=np.array([[r['predicted_full'] for r in grouped[q]] for q in ids]);dot=np.array([[r['predicted_dot_only'] for r in grouped[q]] for q in ids])
    rng=np.random.default_rng(1729);draw=rng.integers(0,32,(10000,32))
    mae_full=abs(full-actual).mean(-1);mae_dot=abs(dot-actual).mean(-1);mae_zero=abs(actual).mean(-1)
    primary=[{'contrast':'MAE_full_minus_dot_only',**estimate(mae_full-mae_dot,draw,2)},
             {'contrast':'MAE_full_minus_zero',**estimate(mae_full-mae_zero,draw,2)}]
    selected=abs(actual)>1e-7
    accuracy={name:float((np.sign(pred[selected])==np.sign(actual[selected])).mean()) for name,pred in [('full',full),('dot_only',dot)]}
    per_unit=[]
    for u in units:
        contrasts=[];raw=[]
        for visit in [1,3]:
            records=[next(r for r in grouped[q] if r['expert']==u['expert'] and r['unit']==u['unit'] and r['visit']==visit) for q in ids]
            values=np.array([r['actual_delta'] for r in records]);raw.append({'visit':visit,**estimate(values,draw)})
            if u['category']=='phase' or (u['expert'],u['unit'])==(45,1127):
                adjusted=values-np.array([np.mean(r['random_deltas']) for r in records]);contrasts.append({'visit':visit,**estimate(adjusted,draw,8 if u['category']=='phase' else 1)})
        reversal=len(contrasts)==2 and ((contrasts[0]['interval'][0]>0 and contrasts[1]['interval'][1]<0) or (contrasts[0]['interval'][1]<0 and contrasts[1]['interval'][0]>0))
        per_unit.append({**u,'raw_effects':raw,'adjusted_effects':contrasts,'passes_sign_reversal_rule':bool(reversal),'is_primary_replication':u['category']=='phase'})
    gain=mae_dot-mae_full;chosen=int(np.argmin(abs(gain-np.median(gain))));q=ids[chosen]
    examples={'selection':'Article nearest the median dot-only-minus-full MAE. Unit/pass with largest error reduction within that article.','id':q,
              'article':grouped[q][0]['article'],'mean_absolute_error_full':float(mae_full[chosen]),'mean_absolute_error_dot_only':float(mae_dot[chosen]),
              'case':grouped[q][int(np.argmax(abs(dot[chosen]-actual[chosen])-abs(full[chosen]-actual[chosen])))]}
    result={'primary_forecast_contrasts':primary,'passes_both_forecast_requirements':all(r['interval'][1]<0 for r in primary),
            'mae':{name:estimate(value,draw) for name,value in [('full',mae_full),('dot_only',mae_dot),('zero',mae_zero)]},
            'direction_accuracy_nonzero_effects':accuracy,'nonzero_effect_cases':int(selected.sum()),'total_unit_pass_article_cases':int(actual.size),
            'per_unit':per_unit,'representative_example':examples,
            'maximum_random_norm_sum_error':max((audit['relative_norm_sum_error'] for r in rows for audit in r['random_audits']),default=0),
            'limits':['All identities were selected on development data. This tests the fixed eight units, not all native units.',
                      'The unit/pass cases are averaged within articles for primary forecasting inference.',
                      'Sign agreement is descriptive. The two forecasting intervals form a separate family from the eight new-unit phase intervals.',
                      'The measured endpoint is a local reader weight, not answer accuracy.']}
    (a.run/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
