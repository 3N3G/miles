"""Preregistered question bootstrap, run only after all three fixed seeds finish."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def stat(value,bootstrap,family=1):
    return {'mean':float(value),'interval':np.quantile(bootstrap,[.025/family,1-.025/family]).tolist(),
            'interval_family_size':family,'resamples':10000,'seed':1729,'sampling_unit':'question after averaging three answer orders'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    spec=json.loads(a.spec.read_text());inputs=json.loads(a.inputs.read_text());expected=[r for r in inputs['records'] if r['cohort']=='confirmation']
    questions={r['id']:r for r in expected};conditions=['clean']+[c['name'] for c in spec['conditions']]
    paired=defaultdict(dict);manifests=[];rows=[]
    for seed in [0,1,2]:
        root=a.root/f'expert-confirmation-s{seed}'
        manifest=json.loads((root/'manifest.json').read_text());assert manifest['status']=='completed'
        assert manifest['spec_sha256']==hashlib.sha256(a.spec.read_bytes()).hexdigest()
        assert manifest['inputs_sha256']==hashlib.sha256(a.inputs.read_bytes()).hexdigest()
        manifests.append(manifest)
        rr=json.loads((root/'results.json').read_text());assert len(rr)==128*len(conditions)
        for r in rr:
            assert r['seed']==seed and r['id'] in questions
            assert np.isfinite(r['gold_choice_log_probability'])
            key=(r['id'],seed);assert r['condition'] not in paired[key]
            paired[key][r['condition']]=r;rows.append(r)
    assert len(paired)==128*3
    for key,value in paired.items():
        assert set(value)==set(conditions)
        assert len({tuple(r['token_ids']) for r in value.values()})==1
        assert len({r['answer'] for r in value.values()})==1
    subjects=['elementary_mathematics','high_school_biology']
    ids={s:[r['id'] for r in expected if r['subject']==s] for s in subjects};assert all(len(v)==64 for v in ids.values())
    rng=np.random.default_rng(1729);draw={s:rng.integers(0,64,size=(10000,64)) for s in subjects}
    def per_question(subject,condition,field):
        return np.array([np.mean([paired[(q,seed)][condition][field] for seed in [0,1,2]]) for q in ids[subject]])
    summary=[];effects={}
    for subject in subjects:
        native=per_question(subject,'clean','gold_choice_log_probability')
        for condition in conditions:
            delta=native-per_question(subject,condition,'gold_choice_log_probability');effects[(subject,condition)]=delta
            acc=per_question(subject,condition,'correct')
            summary.append({'subject':subject,'condition':condition,'questions':64,'answer_order_views':192,
                            'correct_views':int(round(acc.sum()*3)),
                            'accuracy':stat(acc.mean(),acc[draw[subject]].mean(-1)),
                            'choice_nll_increase':stat(delta.mean(),delta[draw[subject]].mean(-1)),
                            'mean_letter_mass':float(per_question(subject,condition,'letter_mass').mean()),
                            'full_vocabulary_nll_increase':float((per_question(subject,'clean','gold_log_probability')-per_question(subject,condition,'gold_log_probability')).mean())})
    primary=[];family=spec['primary_interval_family_size'];examples=[]
    for target in spec['primary_identities']:
        deleted=target['deleted'];random=target['random'];math,bio=subjects
        d=effects[(math,deleted)];c_math=d-effects[(math,random)];c_bio=effects[(bio,deleted)]-effects[(bio,random)]
        direct=stat(d.mean(),d[draw[math]].mean(-1),family)
        above=stat(c_math.mean(),c_math[draw[math]].mean(-1),family)
        difference=stat(c_math.mean()-c_bio.mean(),c_math[draw[math]].mean(-1)-c_bio[draw[bio]].mean(-1),family)
        primary.append({'name':target['name'],'math_native_deletion_harm':direct,'math_harm_above_random':above,
                        'math_minus_biology_harm_above_random':difference,
                        'passes_all_three_preregistered_requirements':all(s['interval'][0]>0 for s in [direct,above,difference])})
        chosen=[('near_median',int(np.argmin(abs(d-np.median(d))))),('largest_positive',int(d.argmax())),('largest_negative',int(d.argmin()))]
        for label,index in chosen:
            q=ids[math][index];native=paired[(q,0)]['clean']
            examples.append({'target':target['name'],'selection':label,'id':q,'mean_three_order_loss_change':float(d[index]),
                             'seed0_full_input':native['rendered_input'],'seed0_gold':native['answer'],
                             'seed0_conditions':{name:{k:paired[(q,0)][name][k] for k in ['prediction','correct','gold_choice_log_probability','gold_log_probability','letter_mass','choice_probabilities','top_tokens']} for name in ['clean',deleted,random]}})
    norm=[]
    random_names={c['name'] for c in spec['conditions'] if c.get('kind')=='matched_random'}
    for name in sorted(random_names):
        relevant=[r for r in rows if r['condition']==name];intended=actual=0.;relative=[]
        for r in relevant:
            x=sum(v['reference_deletion_norm_sum'] for v in r['audit']);y=sum(v['actual_edit_norm_sum'] for v in r['audit'])
            intended+=x;actual+=y
            if x>0:relative.append(abs(y-x)/x)
        norm.append({'condition':name,'intended_total_norm':intended,'actual_total_norm':actual,
                     'aggregate_relative_norm_error':abs(actual-intended)/max(intended,1e-20),
                     'question_relative_norm_error_max':max(relative,default=0.),'question_relative_norm_error_95th':float(np.quantile(relative,.95)) if relative else 0.})
    majority=spec['baselines']['calibration_majority_letter'];baseline=[]
    for subject in subjects:
        correct=sum(paired[(q,seed)]['clean']['answer']==majority for q in ids[subject] for seed in [0,1,2])
        baseline.append({'subject':subject,'constant_calibration_majority_letter':majority,'correct_views':correct,'views':192,'uniform_expected_accuracy':.25,'uniform_choice_nll':float(np.log(4))})
    result={'primary':primary,'descriptive_conditions':summary,'dose_checks':norm,'baselines':baseline,'raw_examples':examples,
            'unique_questions':128,'orders_per_question':3,'source_manifests':manifests,
            'limits':['This is a two-subject native-model test, not a proof of a unique arithmetic algorithm.',
                      'Three order seeds are averaged within each question; they are not extra independent observations.',
                      'Potential MMLU pretraining exposure limits fresh capability claims.',
                      'All nonprimary comparisons have descriptive unadjusted95% intervals.']}
    (a.out/'confirmation_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'primary':primary,'dose_checks':norm,'baselines':baseline},indent=2))


if __name__=='__main__':main()
