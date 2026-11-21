"""Summarize intervention horizons and show all four recovery traces."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
 p=argparse.ArgumentParser();p.add_argument('--matched',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();results=[]
 for dirname,label in [('phase-transfer-v1','third_pass_row'),('early-phase-transfer-v1','own_horizon_row')]:
  run=Path('runs')/dirname;assert (run/'manifest.json').exists();rows=json.loads((run/'interventions.json').read_text())
  for site in sorted({x['site'] for x in rows}):
   for seed in [0,1,2]:
    for dose in [1,2]:
     selected=[x for x in rows if x['site']==site and x['fit_seed']==seed and x['dose']==dose];real=[x for x in selected if x['random_seed'] is None];random=[x for x in selected if x['random_seed'] is not None]
     results.append({'row_source':label,'site':site,'fit_seed':seed,'dose':dose,'cases':len(real),'swaps':sum(x['counterfactual_top1'] for x in real),'swap_names':[x['name'] for x in real if x['counterfactual_top1']],'random_swaps':sum(x['counterfactual_top1'] for x in random),'random_trials':len(random)})
 trace=json.loads(Path('runs/country-recovery-trace-v1/traces.json').read_text());names=list(dict.fromkeys(x['name'] for x in trace));recovery=[]
 for name in names:
  clean=next(x for x in trace if x['name']==name and x['condition']=='clean');edit=next(x for x in trace if x['name']==name and x['condition']=='third_v11')
  recovery.append({'name':name,'original_probe_cosines':[x['probe_cosine_after'] for x in clean['trace']],'edited_probe_cosines':[x['probe_cosine_after'] for x in edit['trace']],'distance_to_clean':[x['normalized_bank_rms_distance_to_clean'] for x in edit['trace']],'original_sign_recovered_by_pass3':np.sign(clean['trace'][-1]['probe_cosine_after'])==np.sign(edit['trace'][-1]['probe_cosine_after']),'native_top1':clean['top10'][0]['text'],'edited_top1':edit['top10'][0]['text'],'target_probability':edit['target_probability']})
 matched=[];rounding=None
 if a.matched:
  assert (a.matched/'manifest.json').exists();rows=json.loads((a.matched/'interventions.json').read_text())
  for method in sorted({x['method'] for x in rows}):
   for seed in [0,1,2]:
    xs=[x for x in rows if x['method']==method and x['fit_seed']==seed];matched.append({'method':method,'fit_seed':seed,'cases':len(xs),'swaps':sum(x['counterfactual_top1'] for x in xs),'swap_names':[x['name'] for x in xs if x['counterfactual_top1']]})
  errors=[]
  for x in rows:
    p=x['perturbation'];expected=np.array(p['expected_norms']);actual=np.array(p['actual_norms']);errors.extend((np.abs(actual-expected)/(expected+1e-6)).tolist())
  rounding={'maximum_precast_norm_match_error':max(x['perturbation']['max_precast_relative_norm_error'] for x in rows),'maximum_jacobian_reference_logit_discrepancy':max(x['jacobian_formula_max_logit_discrepancy'] or 0 for x in rows),'median_actual_BF16_norm_relative_error':float(np.median(errors)),'maximum_actual_BF16_norm_relative_error':float(np.max(errors))}
 result={'scope':'Four related native-valid capital prompts, not a broad intervention success-rate estimate. Fit subsets overlap.','single_site_results':results,'early_swap_recovery':recovery,'matched_size_directions':matched,'rounding_controls':rounding}
 # Convert scalar numpy booleans for auditable JSON.
 for x in recovery:x['original_sign_recovered_by_pass3']=bool(x['original_sign_recovered_by_pass3'])
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
 for row_source,color,shift,label in [('third_pass_row','#0072B2',-.17,'Third-pass direction'),('own_horizon_row','#D55E00',.17,'Separate early-pass fit')]:
  for i,site in enumerate([11,23,35]):
   xs=[x['swaps'] for x in results if x['row_source']==row_source and x['site']==site and x['dose']==2]
   if not xs:continue
   ax.bar(i+shift,np.mean(xs),width=.32,color=color,alpha=.7,label=label if i==0 else None);ax.scatter(i+shift+np.array([-.06,0,.06]),xs,c='black',s=16,zorder=3)
 ax.set_xticks([0,1,2],['Pass 1','Pass 2','Pass 3']);ax.set_ylim(0,4.5);ax.set_yticks(range(5));ax.set_ylabel('Capital answers changed as intended (out of four)');ax.set_title('The same readable direction has different causal effects by pass');ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 fig.text(.02,-.055,'One intervention site at the same physical layer, dose two, all prompt positions.\nDots show three calibration fits. Matched-norm random directions produced no swaps.',fontsize=10)
 fig.savefig('figures/country_single_pass.png',dpi=180,bbox_inches='tight');plt.close(fig)
 titles={'germany-explicit-capital':'Germany named','germany-unspoken-capital':'Germany via Brandenburg Gate','china-explicit-capital':'China named','china-unspoken-capital':'China via Great Wall'}
 fig,axes=plt.subplots(2,2,figsize=(9,6),sharex=True,sharey=True,layout='constrained')
 for ax,r in zip(axes.flat,recovery):
  sign=1 if r['name'].startswith('germany') else -1
  ax.plot([1,2,3],np.array(r['original_probe_cosines'])*sign,'o-',c='#777777',label='Clean')
  ax.plot([1,2,3],np.array(r['edited_probe_cosines'])*sign,'o-',c='#0072B2',label='Edited in pass 1')
  ax.axhline(0,c='black',lw=.8);ax.set_title(titles[r['name']],fontsize=11);ax.set_xticks([1,2,3]);ax.grid(alpha=.2);ax.set_ylim(-.3,.2)
 axes[0,0].legend(fontsize=9);fig.supxlabel('Loop pass');fig.supylabel('Country contrast cosine; positive means original country')
 fig.suptitle('An early edit flips the readable country without changing the final capital',fontsize=13)
 fig.text(.02,-.04,'The original sign returns by pass three in 3/4 cases. All four retain the original answer.\nValues use one fixed third-pass probe. These are the three production states, not a convergence fit.',fontsize=10)
 fig.savefig('figures/country_recovery.png',dpi=180,bbox_inches='tight');plt.close(fig)
 print('single-site',[(s,[x['swaps'] for x in results if x['row_source']=='third_pass_row' and x['site']==s and x['dose']==2]) for s in [11,23,35]])
 print('probe original sign recovered',sum(x['original_sign_recovered_by_pass3'] for x in recovery),'/',len(recovery))
 print('matched',matched)
if __name__=='__main__':main()
