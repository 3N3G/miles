"""Paired article bootstrap for the preregistered native routing sign test."""

import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np


def interval(values,draw,family=1):
    values=np.asarray(values)
    return {'mean':float(values.mean()),'interval':np.quantile(values[draw].mean(-1),[.025/family,1-.025/family]).tolist(),
            'family_size':family,'n_articles':len(values),'sampling_unit':'article','resamples':10000,'seed':1729}


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    manifest=json.loads((a.run/'manifest.json').read_text());assert manifest['status']=='completed'
    rows=json.loads((a.run/'results.json').read_text());paired=defaultdict(dict)
    for row in rows:
        assert row['condition'] not in paired[row['id']];paired[row['id']][row['condition']]=row
    assert len(paired)==16
    rng=np.random.default_rng(1729);draw=rng.integers(0,16,size=(10000,16));primary=[];c=[]
    for visit in [1,3]:
        adjusted=[]
        for q,conditions in sorted(paired.items()):
            row=conditions[f'unit45-pass{visit}'];native=row['native_reader_last_key_mean'][visit-1]
            deleted=row['reader_last_key_mean'][visit-1]-native
            random=np.mean([conditions[f'unit45-pass{visit}-random{s}']['reader_last_key_mean'][visit-1]-native for s in [0,1,2]])
            adjusted.append(deleted-random)
        c.append(np.array(adjusted));primary.append({'metric':f'pass{visit}_block2_weight_change_above_random',**interval(adjusted,draw,3)})
    primary.append({'metric':'pass3_minus_pass1_adjusted_change',**interval(c[1]-c[0],draw,3)})
    lo1,hi1=primary[0]['interval'];lo3,hi3=primary[1]['interval'];reversal=(lo1>0 and hi3<0) or (hi1<0 and lo3>0)
    descriptive=[]
    for condition in sorted(next(iter(paired.values()))):
        records=[v[condition] for q,v in sorted(paired.items())]
        visits=records[0]['visits']
        native=np.array([r['native_reader_last_key_mean'] for r in records]);edited=np.array([r['reader_last_key_mean'] for r in records]);diff=edited-native
        nll=[np.mean(np.array(r['nll'])-r['native_nll']) for r in records]
        intended=sum(v['intended_norm_sum'] for r in records for v in r['audit']);actual=sum(v['actual_norm_sum'] for r in records for v in r['audit'])
        descriptive.append({'condition':condition,'reader_delta_per_pass':[interval(diff[:,j],draw) for j in range(3)],
                            'next_token_nll_increase':interval(nll,draw),'mean_kl':float(np.mean([np.mean(r['kl']) for r in records])),
                            'mean_top1_agreement':float(np.mean([r['native_top1_agreement'] for r in records])),
                            'aggregate_norm_relative_error':abs(actual-intended)/max(intended,1e-20)})
    unit_values=defaultdict(dict);fit_counts=[]
    for q in sorted(paired):
        values=json.loads((a.run/(q+'-native_units.json')).read_text());z={}
        for row in values:
            if row['expert']==45 and row['unit']==1127:
                z[row['visit']]=dict(zip(row['positions'],row['weighted_activation']))
        both=sorted(set(z[1])&set(z[3])&set(range(16,128)))
        early=np.array([z[1][p] for p in both]);late=np.array([z[3][p] for p in both])
        fit_counts.append({'id':q,'paired_selected_positions':len(both),'negative_to_positive':int(((early<0)&(late>0)).sum()),
                           'positive_to_negative':int(((early>0)&(late<0)).sum()),'mean_early':float(early.mean()) if len(early) else None,
                           'mean_late':float(late.mean()) if len(late) else None})
    result={'primary':primary,'passes_preregistered_sign_reversal_rule':bool(reversal),'descriptive':descriptive,'native_unit_replication':fit_counts,
            'article_interactions':[{'id':q,'adjusted_pass1':float(c[0][i]),'adjusted_pass3':float(c[1][i]),'interaction':float(c[1][i]-c[0][i])} for i,q in enumerate(sorted(paired))],
            'limits':['Native local block routing is the tested outcome. This does not establish a global iteration counter.',
                      'All comparisons except the three primary intervals are descriptive, unadjusted95% intervals.']}
    (a.run/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'primary':primary,'sign_reversal':reversal},indent=2))


if __name__=='__main__':main()
