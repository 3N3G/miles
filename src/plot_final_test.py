"""Show paired held-out item outcomes and the frozen uncertainty interval."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

summary=json.load(open('evidence/transport_final_summary.json'));primary=next(x for x in summary['results'] if x['seed']==0 and x['filtered'] and x['subset']=='capable_unspoken')
rows=json.load(open('runs/transport-final-v1/readouts.json'));outputs=json.load(open('runs/transport-final-v1/development_outputs.json'))
selected=[x for x in rows if x['seed']==0 and x['filtered'] and x['eligible'] and x['capable_unspoken']]
items=[x['item'] for x in outputs if any(r['name']==x['item']['name'] for r in selected)]
names=[x['name'] for x in items];grid=np.array([[any(x['hit'] for x in selected if x['name']==n and x['method']==m) for m in ['raw','mean_scale']] for n in names],dtype=int)
fig,axes=plt.subplots(1,2,figsize=(10,8),gridspec_kw={'width_ratios':[1,1.2]});fig.subplots_adjust(left=.14,right=.98,top=.90,bottom=.20,wspace=.18);ax=axes[0]
ax.imshow(grid,cmap=ListedColormap(['#eeeeee','#0072B2']),vmin=0,vmax=1,aspect='auto');ax.set_yticks(range(len(items)),[' / '.join(x['intermediates']) for x in items],fontsize=10);ax.set_xticks([0,1],['Raw','Mean/scale']);ax.xaxis.tick_top();ax.set_xticks(np.arange(-.5,2,1),minor=True);ax.set_yticks(np.arange(-.5,len(items),1),minor=True);ax.grid(which='minor',color='white',lw=1.5);ax.tick_params(which='minor',bottom=False,left=False)
for i,row in enumerate(grid):
 for j,value in enumerate(row):ax.text(j,i,str(value),ha='center',va='center',c='white' if value else '#999999',fontsize=9)
ax.set_ylabel('Unspoken intermediate, in reserved test order');ax.set_xlabel('Blue = recovered in the fixed read budget')
ax=axes[1];point=primary['difference']*100;lo,hi=np.array(primary['paired_bootstrap_95'])*100
ax.axvline(0,c='#777777',ls='--');ax.errorbar([point],[0],xerr=[[point-lo],[hi-point]],fmt='o',color='#0072B2',capsize=8,markersize=8,lw=2);ax.set_xlim(-30,35);ax.set_ylim(-1,1);ax.set_yticks([]);ax.set_xlabel('Corrected minus raw recovery (percentage points)');ax.text(.5,.78,'10/23 versus 9/23',transform=ax.transAxes,ha='center',fontsize=15);ax.text(.5,.68,'Three wins and two losses',transform=ax.transAxes,ha='center',fontsize=11);ax.text(.5,.30,'95% interval: −13.0 to +21.7 points\nExact paired test: p = 1.0',transform=ax.transAxes,ha='center',fontsize=11);ax.grid(axis='x',alpha=.2)
fig.suptitle('Mean and scale calibration did not establish a final-test gain',fontsize=14)
fig.text(.015,.035,'Inkling 2 Mini. 23 of 43 reserved prompts met the single-token, solved, unspoken criteria.\nFour visits × ten tokens, final prompt position, identical vocabulary filter. Seed-zero primary test.\nError bar: 95% paired bootstrap over prompts, 10,000 resamples. The other two fits gave the same counts.',fontsize=10)
fig.savefig('figures/final_bridge_test.png',dpi=180,bbox_inches='tight');fig.savefig('figures/final_bridge_test.pdf',bbox_inches='tight')
