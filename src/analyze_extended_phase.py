"""Article-level preregistered tests of extended native unit trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_unit_atlas import estimate


def normalized_distances(bits):
    """Decode stored BF16 exactly and compare whole token/block states by lag."""
    assert bits.dtype==np.uint16 and bits.shape[0]==24
    states=(bits.astype(np.uint32)<<16).view(np.float32)
    assert np.isfinite(states).all()
    state=states/np.sqrt(np.mean(states**2,axis=-1,keepdims=True)+1e-6)
    values={}
    for lag in [1,2,3,6]:
        values[lag]=np.r_[np.full(lag,np.nan),np.sqrt(np.mean((state[lag:]-state[:-lag])**2,axis=(1,2,3)))]
    return values


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    manifest=json.loads((a.run/'manifest.json').read_text());assert manifest['status']=='completed'
    rows=json.loads((a.run/'results.json').read_text());inputs=json.loads((a.run/'frozen_inputs.json').read_text())
    spec=json.loads((a.run/'frozen_units.json').read_text());units=spec['candidates']+spec['known_comparators']
    ids=[r['id'] for r in inputs['records']];assert len(ids)==16 and len(rows)==16*8*24
    actual=np.empty((16,8,24));full=np.empty_like(actual);dot=np.empty_like(actual)
    lookup={(r['id'],r['expert'],r['unit'],r['visit']):r for r in rows};assert len(lookup)==len(rows)
    for i,q in enumerate(ids):
        for j,u in enumerate(units):
            for k in range(24):
                row=lookup[q,u['expert'],u['unit'],k+1]
                actual[i,j,k]=row['actual_delta'];full[i,j,k]=row['predicted_full'];dot[i,j,k]=row['predicted_dot_only']
    distances={lag:[] for lag in [1,2,3,6]}
    for q in ids:
        with np.load(a.run/(q+'-states.npz')) as stored:ds=normalized_distances(stored['states_bf16_bits'])
        for lag,v in ds.items():distances[lag].append(v)
    distances={lag:np.array(v) for lag,v in distances.items()}
    tail=np.arange(18,24);new=np.array([1]+list(range(3,24)))
    rng=np.random.default_rng(1729);draw=rng.integers(0,16,(10000,16))
    causal_distance={lag:abs(actual[:,:,tail]-actual[:,:,tail-lag]).mean((1,2)) for lag in [1,2,3,6]}
    errors={name:abs(pred[:,:,new]-actual[:,:,new]).mean((1,2)) for name,pred in [('full',full),('dot_only',dot),('zero',np.zeros_like(actual))]}
    contrasts=[('late_state_lag3_minus_lag1',distances[3][:,tail].mean(-1)-distances[1][:,tail].mean(-1)),
               ('late_causal_lag3_minus_lag1',causal_distance[3]-causal_distance[1]),
               ('MAE_full_minus_dot_only_new_visits',errors['full']-errors['dot_only']),
               ('MAE_full_minus_zero_new_visits',errors['full']-errors['zero'])]
    primary=[{'contrast':name,**estimate(value,draw,4)} for name,value in contrasts]
    unit_rows=[]
    for j,u in enumerate(units):
        unit_rows.append({**u,'exact_effect_by_visit':[estimate(actual[:,j,k],draw) for k in range(24)],
                          'late_mean_absolute_effect':estimate(abs(actual[:,j,tail]).mean(-1),draw),
                          'late_lag_absolute_difference':{str(lag):estimate(abs(actual[:,j,tail]-actual[:,j,tail-lag]).mean(-1),draw) for lag in [1,2,3,6]}})
    first=inputs['records'][0]
    result={'article_count':16,'passes':24,'primary_contrasts':primary,
            'passes_recurring_causal_phase_rule':all(r['interval'][1]<0 for r in primary[:2]),
            'passes_new_visit_forecast_rule':all(r['interval'][1]<0 for r in primary[2:]),
            'late_state_distances':{str(lag):estimate(v[:,tail].mean(-1),draw) for lag,v in distances.items()},
            'state_distance_by_visit':{str(lag):[None if k<lag else estimate(v[:,k],draw) for k in range(24)] for lag,v in distances.items()},
            'late_causal_distances':{str(lag):estimate(v,draw) for lag,v in causal_distance.items()},
            'mae_new_visits':{name:estimate(v,draw) for name,v in errors.items()},
            'new_visit_numbers':(new+1).tolist(),'per_unit':unit_rows,
            'first_article_example':{'article':first['article'],'prompt':first['rendered_input'],
                'unit45_1127_effects':[lookup[first['id'],45,1127,k+1]['actual_delta'] for k in range(24)]},
            'limits':['The extension exceeds the checkpoint training horizon of three passes.',
                      'The test concerns eight fixed units at one physical layer, not all expert units.',
                      'The endpoint is the immediate Block 2 reader weight, not token recovery or answer quality.',
                      'Finite approximate recurrence is not proof of a stable limit cycle.']}
    (a.run/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['primary_contrasts','late_state_distances','late_causal_distances','mae_new_visits','passes_recurring_causal_phase_rule','passes_new_visit_forecast_rule']},indent=2))


if __name__=='__main__':main()
