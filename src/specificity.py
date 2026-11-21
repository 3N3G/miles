"""Development-only gold-versus-same-type-distractor readout diagnostic."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--readouts',type=Path,required=True);p.add_argument('--tokens',type=Path,required=True);p.add_argument('--items',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rows=json.loads(a.readouts.read_text());tokens=json.loads(a.tokens.read_text());vocab={x.strip().casefold() for x in tokens};items={x['name']:x for x in json.loads(a.items.read_text())['items']}
    for row in rows:
        for token in row['top10']:assert tokens[token['id']]==token['text'],token
    all_results=[]
    for filtered in [True,False]:
      for seed in [0,1,2]:
        scores={};details={}
        methods=sorted({x['method'] for x in rows if x['mode']=='active'})
        for method in methods:
            selected=[x for x in rows if x['mode']=='active' and x['seed']==seed and x['filtered']==filtered and x['method']==method and x['eligible'] and x['capable_unspoken']]
            byname={}
            for r in selected:byname.setdefault(r['name'],[]).append(r)
            scores[method]={};details[method]=[]
            for name,rs in byname.items():
                distractors={x for x in items[name].get('mc',{}).get('distractors',[]) if x.casefold() in vocab}
                if not distractors:continue
                found=set().union(*(set(x['distractors_read']) for x in rs));gold=any(x['hit'] for x in rs)
                other=len(found)/len(distractors);score=float(gold)-other;scores[method][name]=score
                details[method].append({'name':name,'prompt':items[name]['prompt'],'intermediates':items[name]['intermediates'],'gold_hit':gold,'representable_distractors':sorted(distractors),'distractors_read':sorted(found),'distractor_fraction':other,'specificity_score':score})
        for method,sc in scores.items():
            names=sorted(sc);ds=details[method];paired=None;ci=None
            if 'raw' in scores:
                common=sorted(set(sc)&set(scores['raw']));x=np.array([sc[n]-scores['raw'][n] for n in common]);paired=float(x.mean());rng=np.random.default_rng(20261121);ci=np.quantile(x[rng.integers(0,len(x),(10000,len(x)))].mean(-1),[.025,.975]).tolist()
            all_results.append({'method':method,'seed':seed,'filtered':filtered,'n':len(names),'gold_hits':sum(x['gold_hit'] for x in ds),'any_distractor_cases':sum(bool(x['distractors_read']) for x in ds),'mean_distractor_fraction':float(np.mean([x['distractor_fraction'] for x in ds])),'mean_specificity_score':float(np.mean(list(sc.values()))),'paired_specificity_difference_vs_raw':paired,'paired_bootstrap_95':ci,'details':ds})
    a.out.write_text(json.dumps({'scope':'Development diagnostic. A listed distractor can be a valid final answer or considered alternative. This is not a definitive false-positive label.','results':all_results},indent=2)+'\n')
    for x in all_results:
      if x['seed']==0 and x['filtered']:print(json.dumps({k:v for k,v in x.items() if k!='details'}))
if __name__=='__main__':main()
