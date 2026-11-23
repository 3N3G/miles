"""Descriptive activity, dose and subgroup checks after the fixed atlas analysis."""

import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();root=a.run
    assert json.loads((root/'manifest.json').read_text())['status']=='completed'
    spec=json.loads((root/'frozen_units.json').read_text());units=spec['candidates']+spec['known_comparators'];profiles=[]
    rows=json.loads((root/'results.json').read_text());inputs=json.loads((root/'frozen_inputs.json').read_text())['records']
    for u in units:
        e,n=u['expert'],u['unit'];paired=[];byvisit={v:[] for v in [1,2,3]};narticles=0
        for path in sorted(root.glob('unit-atlas-test-*-units.json')):
            selected=[r for r in json.loads(path.read_text()) if r['expert']==e and r['unit']==n];z={}
            for r in selected:
                z[r['visit']]={pos:(h,g,up) for pos,h,g,up in zip(r['positions'],r['weighted_activation'],r['raw_gate'],r['raw_up']) if pos>=16}
                byvisit[r['visit']].extend(z[r['visit']].values())
            both=sorted(set(z.get(1,{}))&set(z.get(3,{})));narticles+=bool(both)
            paired.extend([*z[1][pos],*z[3][pos]] for pos in both)
        values=np.asarray(paired);flips=(values[:,0]*values[:,3]<0) if len(values) else np.array([],bool)
        profiles.append({'expert':e,'unit':n,'category':u['category'],'paired_positions':len(values),'paired_articles':narticles,
            'means_weighted_h_gate_up_per_pass':{v:np.mean(byvisit[v],0).tolist() if byvisit[v] else None for v in [1,2,3]},
            'paired_mean_early_late_h':values[:,[0,3]].mean(0).tolist() if len(values) else None,
            'negative_to_positive':int(((values[:,0]<0)&(values[:,3]>0)).sum()) if len(values) else 0,
            'positive_to_negative':int(((values[:,0]>0)&(values[:,3]<0)).sum()) if len(values) else 0,
            'gate_sign_changes_among_output_flips':int((flips&(values[:,1]*values[:,4]<0)).sum()) if len(values) else 0,
            'up_sign_changes_among_output_flips':int((flips&(values[:,2]*values[:,5]<0)).sum()) if len(values) else 0})
    (root/'unit_activity_profiles.json').write_text(json.dumps({'status':'Descriptive state observations on the completed fixed test, not additional independent tests.','units':profiles},indent=2)+'\n')
    audits=[]
    for u in units:
        for visit in [1,3]:
            entries=[v for r in rows if r['expert']==u['expert'] and r['unit']==u['unit'] and r['visit']==visit for v in r['random_audits']]
            if not entries:continue
            intended=sum(v['intended_norm_sum'] for v in entries);actual=sum(v['actual_norm_sum'] for v in entries)
            audits.append({'expert':u['expert'],'unit':u['unit'],'visit':visit,'aggregate_norm_relative_error':abs(actual-intended)/max(intended,1e-20),'max_article_norm_relative_error':max(v['relative_norm_sum_error'] for v in entries)})
    (root/'dose_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    groups=[]
    for name,condition in [('six_new',lambda r:r['category']!='known'),('two_known',lambda r:r['category']=='known')]:
        z=[r for r in rows if condition(r)];mae={k:float(np.mean([abs(r[k]-r['actual_delta']) for r in z])) for k in ['predicted_full','predicted_dot_only']}
        mae['zero']=float(np.mean([abs(r['actual_delta']) for r in z]));groups.append({'group':name,'cases':len(z),'mae':mae})
    (root/'forecast_subgroups.json').write_text(json.dumps({'scope':'Descriptive checks. The primary unit set and two primary comparisons are unchanged.','groups':groups},indent=2)+'\n')
    articles=[]
    for record in inputs:
        z=[r for r in rows if r['id']==record['id']]
        full=np.array([abs(r['predicted_full']-r['actual_delta']) for r in z]);dot=np.array([abs(r['predicted_dot_only']-r['actual_delta']) for r in z])
        articles.append({'id':record['id'],'article':record['article'],'full_input':record['rendered_input'],
                         'mean_error_reduction':float((dot-full).mean()),'largest_case_error':float(full.max()),
                         'largest_case_improvement':z[int((dot-full).argmax())],'largest_error_case':z[int(full.argmax())]})
    gain=np.array([r['mean_error_reduction'] for r in articles]);chosen=[('nearest_median_gain',int(np.argmin(abs(gain-np.median(gain))))),('smallest_gain',int(gain.argmin())),('largest_gain',int(gain.argmax())),('largest_absolute_prediction_error',int(np.argmax([r['largest_case_error'] for r in articles])))]
    examples=[{'selection':label,**articles[i]} for label,i in chosen]
    (root/'forecast_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    print(json.dumps({'subgroups':groups,'examples':[{k:r[k] for k in ['selection','id','article','mean_error_reduction','largest_case_error']} for r in examples]},indent=2))


if __name__=='__main__':main()
