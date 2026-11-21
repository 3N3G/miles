"""Full-state lag diagnostics and independently evaluated shared phase patterns."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ci_mean(x):
    x=np.asarray(x);rng=np.random.default_rng(20261121)
    return np.quantile(x[rng.integers(0,len(x),(10000,len(x)))].mean(-1),[.025,.975]).tolist()


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert (a.run/'manifest.json').exists()
 confirm=(a.run/'confirmation.json').exists();records=json.loads((a.run/('confirmation.json' if confirm else 'trajectories.json')).read_text());n=len(records);count=64 if confirm else 32;begin=48 if confirm else 16
 lags=[];series=[];lasts=[];examples=[]
 for i,row in enumerate(records):
    state=np.load(a.run/f'states_{i:02}.npz')['states'].astype(np.float32)
    z=state/np.sqrt(np.mean(state*state,axis=-1,keepdims=True)+1e-6)
    bylag=[]
    for lag in range(1,9):bylag.append(float(np.sqrt(np.mean((z[begin:]-z[begin-lag:count-lag])**2,axis=(1,2,3))).mean()))
    lags.append(bylag);series.append({lag:np.sqrt(np.mean((z[lag:]-z[:-lag])**2,axis=(1,2,3))).tolist() for lag in [1,3]});lasts.append(z[:,:,-1].reshape(count,-1))
    examples.append({'index':i,'bank_two_coordinate_2027':z[:,1,-1,2027].tolist(),'lag_distances':bylag})
 lags=np.array(lags);ratios=lags[:,2]/lags[:,0];lasts=np.array(lasts)
 tail=np.arange(begin,count);phase=(tail+1)%3
 phase_means=np.stack([lasts[:,tail[phase==k]].mean(1) for k in range(3)],axis=1);phase_means-=phase_means.mean(1,keepdims=True)
 shared=[]
 for i in range(n):
    mu=np.delete(phase_means,i,axis=0).mean(0);_,_,vh=np.linalg.svd(mu,full_matrices=False);q=vh[:2];test=phase_means[i]
    explained=float(np.sum((test@q.T)**2)/np.sum(test**2))
    z=lasts[i,tail];corrected=z-mu[phase]
    # Compare adjacent tail states, using the same sample range before and after correction.
    d0=np.sqrt(np.mean((z[1:]-z[:-1])**2,axis=-1)).mean();d1=np.sqrt(np.mean((corrected[1:]-corrected[:-1])**2,axis=-1)).mean()
    shared.append({'index':i,'leave_one_paragraph_out_rank_two_phase_variance_explained':explained,'last_token_one_step_before_template_subtraction':float(d0),'last_token_one_step_after_template_subtraction':float(d1)})
 result={'scope':'Independent-text confirmation' if confirm else 'Exploratory development prompt extension','paragraphs':n,'passes':count,'tail_passes':[begin+1,count],'lag_mean_distances':lags.mean(0).tolist(),'lag_mean_bootstrap_95':[ci_mean(lags[:,j]) for j in range(8)],'lag_three_to_one_ratio_by_paragraph':ratios.tolist(),'mean_ratio':float(ratios.mean()),'mean_ratio_bootstrap_95':ci_mean(ratios),'all_native_head_checks_exact':all(all(v==0 for v in r['native_head_errors']) for r in records) if confirm else all(v['native_head_error']==0 for r in records for v in r['stops'].values()),'first_three_states_exact':all(r['first_three_states_exact'] for r in records),'shared_phase_diagnostic':shared,'examples':examples}
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['examples','shared_phase_diagnostic','lag_mean_bootstrap_95']}))
 Path('figures').mkdir(exist_ok=True);prefix='period_confirm' if confirm else 'period_exploratory'
 fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
 for i in range(n):
    for lag,color in [(1,'#777777'),(3,'#0072B2')]:ax.plot(np.arange(lag+1,count+1),series[i][lag],color=color,alpha=.3,lw=1)
 for lag,color,label in [(1,'#777777','One pass apart'),(3,'#0072B2','Three passes apart')]:
    values=np.array([s[lag] for s in series]);ax.plot(np.arange(lag+1,count+1),values.mean(0),color=color,lw=2.5,label=label)
 ax.set_ylim(0,1.15);ax.set_xlim(1,count);ax.set_xlabel('Completed loop pass');ax.set_ylabel('RMS distance between normalized full states');ax.set_title('The recurrent state returns closer after three passes');ax.legend();ax.grid(alpha=.2)
 fig.text(.02,-.055,f'{n} separate '+('WikiText paragraphs' if confirm else 'development prompts')+'. Both recurrent banks, every token position.\nThin lines show individual inputs. The production checkpoint normally runs three passes.',fontsize=10)
 fig.savefig(f'figures/{prefix}_trajectory.png',dpi=180,bbox_inches='tight');fig.savefig(f'figures/{prefix}_trajectory.pdf',bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');means=lags.mean(0);cis=np.array([ci_mean(lags[:,j]) for j in range(8)])
 ax.bar(np.arange(1,9),means,color=['#0072B2' if j in [3,6] else '#999999' for j in range(1,9)],alpha=.7)
 ax.errorbar(np.arange(1,9),means,yerr=np.array([means-cis[:,0],cis[:,1]-means]),fmt='none',color='black',capsize=4)
 for i in range(n):ax.scatter(np.arange(1,9)+(i-(n-1)/2)*.018,lags[i],s=10,color='black',alpha=.5,zorder=3)
 ax.set_ylim(0,1.05);ax.set_xticks(range(1,9));ax.set_xlabel('Pass separation (lag)');ax.set_ylabel('Mean normalized full-state RMS distance');ax.set_title('Lag three and lag six reveal an approximate three-pass orbit');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 fig.text(.02,-.045,f'Passes {begin+1}–{count}; {n} inputs. Dots show inputs; bars show means.\nError bars: 95% bootstrap intervals over inputs, not token rows.',fontsize=10)
 fig.savefig(f'figures/{prefix}_lags.png',dpi=180,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
