"""Two standalone figures for the fixed native-unit forecast test."""

import argparse,json
from pathlib import Path
import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();result=json.loads((a.run/'analysis.json').read_text())
    methods=['full','dot_only','zero'];means=np.array([result['mae'][m]['mean'] for m in methods])*100
    bounds=np.array([result['mae'][m]['interval'] for m in methods])*100
    fig,ax=plt.subplots(figsize=(7.4,4.8));x=np.arange(3)
    ax.bar(x,means,color=['#222222','#0072B2','#999999'],width=.6)
    ax.errorbar(x,means,yerr=np.stack([means-bounds[:,0],bounds[:,1]-means]),fmt='none',ecolor='#555555',capsize=4)
    ax.set(xticks=x,xticklabels=['With normalization','Dot product only','Predict zero change'],ylabel='Mean absolute prediction error (percentage points)',ylim=(0,max(bounds[:,1])*1.15),title='Predicting native unit deletion effects')
    fig.subplots_adjust(left=.16,bottom=.28)
    fig.text(.16,.035,'Inkling 2 Mini: layer 12 units → layer 13 Block 2 read weight.\nEight fixed units, passes 1 and 3. Errors average within each article.\n32 unused articles. Descriptive 95% article-bootstrap intervals.\nPrimary paired comparisons use a separate two-test correction.',fontsize=9)
    fig.savefig('figures/native_unit_atlas_forecast.png',dpi=180);fig.savefig('figures/native_unit_atlas_forecast.pdf');plt.close(fig)
    units=[u for u in result['per_unit'] if u['is_primary_replication']]
    fig,ax=plt.subplots(figsize=(7.4,5.1));y=np.arange(4)
    for visit,offset,color,label in [(1,-.12,'#0072B2','Pass 1'),(3,.12,'#D55E00','Pass 3')]:
        estimates=[next(r for r in u['adjusted_effects'] if r['visit']==visit) for u in units]
        means=np.array([r['mean'] for r in estimates])*100;bounds=np.array([r['interval'] for r in estimates])*100
        ax.errorbar(means,y+offset,xerr=np.stack([means-bounds[:,0],bounds[:,1]-means]),fmt='o',color=color,capsize=4,label=label,markersize=6)
    ax.axvline(0,color='#777777',lw=.9);ax.invert_yaxis()
    ax.set(yticks=y,yticklabels=[f"Expert {u['expert']}, unit {u['unit']}" for u in units],xlabel='Block 2 read-weight change (percentage points)',title='Testing four new phase-sensitive native units')
    ax.legend(frameon=False,loc='best');fig.subplots_adjust(left=.29,bottom=.28)
    fig.text(.13,.035,'Layer 12 unit deletion minus three random edits with matching output norms.\nIdentities selected on separate development articles before this test.\n32 unused articles. Article-bootstrap intervals correct for eight tests.\nA sign reversal requires both intervals to exclude zero in opposite directions.',fontsize=9)
    fig.savefig('figures/native_unit_atlas_phase.png',dpi=180);fig.savefig('figures/native_unit_atlas_phase.pdf');plt.close(fig)


if __name__=='__main__':main()
