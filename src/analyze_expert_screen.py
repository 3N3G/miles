"""Article/question-level effects. Keep association and intervention results distinct."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def interval(values, family=1):
    values=np.asarray(values,dtype=float)
    assert np.isfinite(values).all() and len(values)>1
    rng=np.random.default_rng(1729)
    means=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(-1)
    return {'mean':float(values.mean()),'interval':np.quantile(means,[.025/family,1-.025/family]).tolist(),
            'n':len(values),'unit':'article or question, not tokens or loop visits','resamples':10000,'seed':1729,'family_size':family}


def load_rows(root,name):
    return [row for path in sorted(root.glob(f'rank*/{name}.json')) for row in json.loads(path.read_text())]


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();root=a.run
    summary={};questions=load_rows(root,'question_results');text=load_rows(root,'text_results')
    paired=defaultdict(dict)
    for row in questions:paired[row['id']][row['condition']]=row
    qsummary=[]
    for subject in sorted({r['subject'] for r in questions}):
        for cond in sorted({r['condition'] for r in questions}):
            rows=[x for x in paired.values() if x['clean']['subject']==subject and cond in x]
            qsummary.append({'subject':subject,'condition':cond,'correct':sum(x[cond]['correct'] for x in rows),
                             'n':len(rows),'mean_letter_mass':float(np.mean([x[cond]['letter_mass'] for x in rows])),
                             'mean_gold_choice_lp':float(np.mean([x[cond]['gold_choice_log_probability'] for x in rows])),
                             'choice_nll_increase':interval([x['clean']['gold_choice_log_probability']-x[cond]['gold_choice_log_probability'] for x in rows])})
    summary['development_questions']=qsummary
    articles=defaultdict(dict)
    for row in text:
        if row['cohort']=='text_confirmation':articles[row['id']][row['condition']]=row
    ts=[];contrasts=[]
    for cond in sorted({r['condition'] for r in text}):
        effects=[];strat=defaultdict(list)
        for key,rows in articles.items():
            if cond not in rows:continue
            clean=rows['clean'];change=np.array(rows[cond]['nll'])-np.array(clean['nll'])
            effects.append(change.mean())
            for field in ['numeric_target','numeric_input']:
                mask=np.array(clean[field],dtype=bool)
                if mask.any() and (~mask).any():
                    strat[field].append(float(change[mask].mean()-change[~mask].mean()))
            if cond=='105-all':
                contrasts.append({'id':key,'article':clean['article'],'mean_nll_increase':float(change.mean()),
                                  'rendered_input':clean['rendered_input'],'token_ids':clean['token_ids'],
                                  'positions':clean['prediction_positions'],'loss_change':change.tolist(),
                                  'numeric_target':clean['numeric_target'],'numeric_input':clean['numeric_input']})
        ts.append({'condition':cond,'mean_nll_increase':interval(effects),
                   'contrasts':{field:interval(v,family=2 if cond=='105-all' else 1) for field,v in strat.items()}})
    summary['prospective_text']=ts;summary['article_examples_105']=sorted(contrasts,key=lambda x:x['mean_nll_increase'])
    routing=[]
    for visit in range(3):
        rows=[]
        for article,data in articles.items():
            clean=data['clean'];path=next(root.glob(f'rank*/{article}.npz'));r=np.load(path)
            hit=(r['indices'][visit,16:127]==105).any(-1)
            cue=np.array(clean['numeric_input'],dtype=bool)
            assert len(hit)==len(cue)
            rows.append([int((hit&cue).sum()),int(cue.sum()),int((hit&~cue).sum()),int((~cue).sum())])
        rows=np.array(rows);sums=rows.sum(0);ratio=(sums[0]/sums[1])/(sums[2]/sums[3])
        rng=np.random.default_rng(1729);samples=rows[rng.integers(0,len(rows),size=(10000,len(rows)))].sum(1)
        ratios=(samples[:,0]/samples[:,1])/(samples[:,2]/samples[:,3])
        routing.append({'visit':visit+1,'selected_numeric':int(sums[0]),'numeric_tokens':int(sums[1]),
                        'selected_other':int(sums[2]),'other_tokens':int(sums[3]),'rate_ratio':float(ratio),
                        'family95_interval':np.quantile(ratios,[.025/3,1-.025/3]).tolist(),'article_count':len(rows)})
    summary['prospective_routing_105']=routing
    grads=load_rows(root,'router_gradients');rank=defaultdict(lambda:defaultdict(list))
    for row in grads:
        allvisits=defaultdict(lambda:np.zeros(128))
        for s in row['scores']:allvisits[s['layer_index']]+=np.array(s['signed'])
        for layer,values in allvisits.items():
            for expert,value in enumerate(values):rank[(layer,expert)][row['subject']].append(float(value))
    ranks=[]
    for (layer,expert),data in rank.items():
        math=float(np.mean(data['elementary_mathematics']));bio=float(np.mean(data['high_school_biology']))
        ranks.append({'layer_index':layer,'expert':expert,'math':math,'biology':bio,'math_minus_biology':math-bio})
    summary['exploratory_gradient_ranking']=sorted(ranks,key=lambda x:x['math_minus_biology'],reverse=True)
    summary['limits']=['Development questions are exploratory. Numeric cue list includes time words. Only text article tests were prospective.',
                       'Routing derivatives fix top-k choices locally and need exact validation.',
                       'Bootstrap is over unique articles/questions. No loop visit is counted as an independent sample.']
    (root/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'questions':qsummary,'text':ts,'routing':routing,'gradient_top10':summary['exploratory_gradient_ranking'][:10]},indent=2))


if __name__=='__main__':main()
