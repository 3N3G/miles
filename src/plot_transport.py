"""Scientific plots with visible sampling units and fixed read budgets."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={'raw':'#777777','mean_scale':'#0072B2','ridge':'#D55E00','direct_rotation':'#CC79A7','recent_rotation_power':'#009E73','projected_fixed_point':'#E69F00'}
LABELS={'raw':'Raw','mean_scale':'Mean/scale','ridge':'State predictor','direct_rotation':'Direct rotation','recent_rotation_power':'Repeated rotation','projected_fixed_point':'Projected fixed point'}


def main():
    Path('figures').mkdir(exist_ok=True)
    summary=json.load(open('evidence/transport_cycle_development.json'))
    averages=[x for x in summary['averages'] if x['mode']=='active' and x['filtered'] and x['method'] in COLORS]
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    offsets={'raw':(6,-4),'mean_scale':(7,2),'ridge':(7,1),'direct_rotation':(7,5),'recent_rotation_power':(7,5),'projected_fixed_point':(-140,10)}
    for row in averages:
        m=row['method'];x=row['mean_validation_normalized_mse']; y=np.mean(row['capable_hits_by_seed'])/row['capable_eligible']
        ax.scatter(x,y,s=65,color=COLORS[m],zorder=4)
        ax.annotate(LABELS[m],(x,y),xytext=offsets[m],textcoords='offset points',fontsize=10,color=COLORS[m])
    ax.set_xscale('log');ax.set_ylim(-.04,1);ax.set_xlim(.55,90)
    ax.set_xlabel('Final-state squared prediction error / final-state variance (log scale)')
    ax.set_ylabel('Unspoken intermediate recovery (fraction of 24 cases)')
    ax.set_title('Better final-state prediction did not give better bridge readout',fontsize=14)
    ax.grid(alpha=.2)
    fig.text(.02,-.045,'Development only. Four fixed visits × ten tokens, identical letter/digit filter.\nPoints average three calibration fits. Geometry uses sixteen separate text paragraphs.',fontsize=10)
    fig.savefig('figures/prediction_vs_bridge.png',dpi=180,bbox_inches='tight');fig.savefig('figures/prediction_vs_bridge.pdf',bbox_inches='tight');plt.close(fig)
    order=['raw','mean_scale','projected_fixed_point','direct_rotation','ridge','recent_rotation_power']
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for i,m in enumerate(order):
        r=next(x for x in averages if x['method']==m);vals=r['capable_hits_by_seed'];ax.bar(i,np.mean(vals),color=COLORS[m],alpha=.65,width=.65)
        ax.scatter(i+np.array([-.13,0,.13]),vals,color='black',s=20,zorder=3)
    ax.set_ylim(0,24);ax.set_ylabel('Recovered unspoken intermediates (out of 24)');ax.set_xticks(range(len(order)),[LABELS[x] for x in order],rotation=20,ha='right');ax.set_title('Simple calibration beat the more complex transport maps')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.02,-.045,'Development only. Same four visits, last prompt token, K=10, identical token filter.\nBlack dots are the three calibration fits, not independent evaluation samples.',fontsize=10)
    fig.savefig('figures/development_bridges.png',dpi=180,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
