"""One figure for the preregistered native routing sign reversal."""

import argparse,json
from pathlib import Path
import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=json.loads((a.run/'analysis.json').read_text())
    means=np.array([v['mean'] for v in r['primary'][:2]])*100
    bounds=np.array([v['interval'] for v in r['primary'][:2]])*100
    fig,ax=plt.subplots(figsize=(7.2,4.8));x=np.array([1,3])
    ax.errorbar(x,means,yerr=np.stack([means-bounds[:,0],bounds[:,1]-means]),fmt='o',color='#0072B2',capsize=5,markersize=7)
    ax.axhline(0,color='#777777',lw=.9)
    ax.set(xticks=[1,3],xticklabels=['Pass 1','Pass 3'],xlim=(.5,3.8),
           ylabel='Change in Block 2 read weight (percentage points)',
           title='The same native unit has opposite routing effects')
    for i,mean in zip(x,means):ax.annotate(f'{mean:+.3f} pp',(i,mean),xytext=(12,0),textcoords='offset points',va='center',fontsize=11)
    fig.subplots_adjust(left=.17,bottom=.30)
    fig.text(.17,.035,'Inkling 2 Mini: layer 12, expert 45, unit 1127. Outcome: layer 13 block reader.\nNative deletion minus the mean of three edits with matched output norms.\n16 unseen articles. Article-bootstrap intervals adjust for three primary tests.\nNative Block 2 read weight: 70.10% on pass 1 and 87.64% on pass 3.',fontsize=9)
    Path('figures').mkdir(exist_ok=True);fig.savefig('figures/native_unit45_causal_routing.png',dpi=180);fig.savefig('figures/native_unit45_causal_routing.pdf');plt.close(fig)


if __name__=='__main__':main()
