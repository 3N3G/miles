"""Frozen paired final-test counts, uncertainty, exclusions, and raw examples."""
import argparse,json,math,re
from pathlib import Path
import numpy as np


def interval(x):
    rng=np.random.default_rng(20261121)
    samples=x[rng.integers(0,len(x),(10000,len(x)))].mean(-1)
    return np.quantile(samples,[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert (a.run/'manifest.json').exists()
    rows=json.loads((a.run/'readouts.json').read_text());native=json.loads((a.run/'development_outputs.json').read_text())
    results=[];examples=[]
    for seed in [0,1,2]:
      for filtered in [True,False]:
       for subset in ['capable_unspoken','all_eligible']:
        selected=[x for x in rows if x['seed']==seed and x['filtered']==filtered and x['eligible'] and (subset=='all_eligible' or x['capable_unspoken'])]
        names=sorted({x['name'] for x in selected})
        hits={m:{n:any(x['hit'] for x in selected if x['method']==m and x['name']==n) for n in names} for m in ['raw','mean_scale']}
        raw=np.array([hits['raw'][n] for n in names]);cal=np.array([hits['mean_scale'][n] for n in names]);diff=cal.astype(float)-raw
        wins=int(np.sum(cal&~raw));losses=int(np.sum(raw&~cal));discord=wins+losses
        exact=min(1.,2*sum(math.comb(discord,k) for k in range(min(wins,losses)+1))/2**discord) if discord else 1.
        results.append({'seed':seed,'filtered':filtered,'subset':subset,'n':len(names),'raw_hits':int(raw.sum()),'corrected_hits':int(cal.sum()),'wins':wins,'losses':losses,'difference':float(diff.mean()) if len(diff) else None,'paired_bootstrap_95':interval(diff) if len(diff) else None,'exact_mcnemar_two_sided':exact,'win_names':[n for n in names if hits['mean_scale'][n] and not hits['raw'][n]],'loss_names':[n for n in names if hits['raw'][n] and not hits['mean_scale'][n]]})
        if seed==0 and filtered and subset=='capable_unspoken':
          unchanged=next((o['item']['name'] for o in native if o['item']['name'] in names and hits['raw'][o['item']['name']]==hits['mean_scale'][o['item']['name']]),None)
          for n in names:
            if hits['raw'][n]==hits['mean_scale'][n] and n!=unchanged: continue
            original=next(x for x in native if x['item']['name']==n)
            examples.append({'name':n,'role':'unchanged control' if n==unchanged else 'win' if hits['mean_scale'][n] else 'loss','prompt':original['item']['prompt'],'intermediates':original['item']['intermediates'],'target':original['item']['target'],'model_continuation':original['generated_text'],'raw_hit':hits['raw'][n],'corrected_hit':hits['mean_scale'][n],'readouts':[{k:v for k,v in x.items() if k in ['method','visit','hit','top10']} for x in selected if x['name']==n]})
    exclusions=[]
    for original in native:
        name=original['item']['name']; row=next(x for x in rows if x['name']==name)
        if row['eligible'] and row['capable_unspoken']: continue
        item=original['item'];reasons=[]
        if not row['eligible']:reasons.append('no complete single-token intermediate')
        if not re.match(r'^[\s\"\x27]*'+re.escape(item['target'])+r'(?!\w)',original['generated_text'],re.I):reasons.append('greedy continuation does not start with source target')
        if any(re.search(r'(?<!\w)'+re.escape(c)+r'(?!\w)',original['generated_text'],re.I) for c in item['intermediates']):reasons.append('intermediate appears in continuation')
        if any(re.search(r'(?<!\w)'+re.escape(c)+r'(?!\w)',item['prompt'],re.I) for c in item['intermediates']):reasons.append('intermediate appears in input')
        exclusions.append({'name':name,'reasons':reasons,'prompt':item['prompt'],'source_target':item['target'],'intermediates':item['intermediates'],'continuation':original['generated_text']})
    result={'scope':'Frozen held-out 43-item test. One primary comparison: seed0, filtered, capable_unspoken. Other views descriptive. Bootstrap resamples cases.','results':results,'paired_examples':examples,'primary_exclusions':exclusions,'all_native_head_checks_exact':all(x['native_logit_error']==0 for x in native)}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    for x in results:
        if x['subset']=='capable_unspoken':print(json.dumps(x))
if __name__=='__main__':main()
