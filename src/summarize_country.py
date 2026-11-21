"""Case-based country-swap summaries with explicit validity gates."""
import argparse,json,math
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();results=[]
    for run in a.runs:
      assert (run/'manifest.json').exists(),run
      rows=json.loads((run/'interventions.json').read_text())
      for method in sorted({x['method'] for x in rows}):
       for seed in [0,1,2]:
        for dose in [.5,1,2]:
         selected=[x for x in rows if x['method']==method and x['fit_seed']==seed and x['dose']==dose and x['kind']!='unrelated' and x['baseline_valid'] and x['counterfactual_input_valid']]
         real=[x for x in selected if x['random_seed'] is None];random=[x for x in selected if x['random_seed'] is not None]
         n=len(real);hits=sum(x['counterfactual_top1'] for x in real)
         if n:
          z=1.95996398454;center=(hits/n+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(hits/n*(1-hits/n)/n+z*z/(4*n*n))/(1+z*z/n);wilson=[center-half,center+half]
         else:wilson=None
         effects=[]
         for x in real:
          controls=[y['counterfactual_top1'] for y in random if y['name']==x['name']]
          if controls: effects.append(float(x['counterfactual_top1'])-float(np.mean(controls)))
         ci=None
         if effects:
          e=np.array(effects);rng=np.random.default_rng(20261121);boot=e[rng.integers(0,len(e),(10000,len(e)))].mean(-1);ci=np.quantile(boot,[.025,.975]).tolist()
         results.append({'run':run.name,'method':method,'fit_seed':seed,'dose':dose,'valid_cases':n,'swaps':hits,'swap_names':[x['name'] for x in real if x['counterfactual_top1']],'wilson_95_descriptive':wilson,'matched_random_trials':len(random),'matched_random_swaps':sum(x['counterfactual_top1'] for x in random),'paired_case_bootstrap_difference_95':ci,'bootstrap_warning':'All-success small samples produce degenerate bootstrap intervals. Four templates are related, not four independent semantic discoveries.','examples':[{k:x[k] for k in ['name','prompt','top5','source_probability','target_probability']} for x in real]})
    a.out.write_text(json.dumps({'scope':'Strict validity requires both original and direct counterfactual native inputs to answer as expected. Fit seeds and random seeds are not additional independent cases.','results':results},indent=2)+'\n')
    for x in results:
      if x['dose']==2:print(json.dumps({k:v for k,v in x.items() if k not in ['examples','bootstrap_warning']}))
if __name__=='__main__':main()
