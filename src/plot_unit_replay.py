"""Show why a direct writer contribution gives the wrong average routing sign."""

import argparse,json
from pathlib import Path
import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    result=json.loads((a.run/'analysis.json').read_text());rows=[r for r in result['effects'] if r['condition']=='unit45']
    fig,ax=plt.subplots(figsize=(7.5,4.9))
    for offset,(key,label,color) in zip([-.15,0,.15],[('numerator_contribution','Dot product','#0072B2'),('rms_contribution','RMS normalization','#D55E00'),('delta_block2_weight','Actual total','#222222')]):
        means=np.array([r[key]['mean'] for r in rows])*100
        bounds=np.array([r[key]['descriptive_95_interval'] for r in rows])*100
        ax.errorbar(np.array([1,2])+offset,means,yerr=np.stack([means-bounds[:,0],bounds[:,1]-means]),fmt='o',color=color,capsize=4,label=label,markersize=6)
    ax.axhline(0,color='#777777',lw=.9)
    ax.set(xticks=[1,2],xticklabels=['Pass 1','Pass 3'],xlim=(.6,2.4),ylabel='Block 2 read-weight change (percentage points)',title='Normalization reverses the direct writer effect')
    ax.legend(frameon=False,loc='upper right',fontsize=9)
    fig.subplots_adjust(left=.16,bottom=.29)
    fig.text(.16,.035,'Inkling 2 Mini, layer 12, expert 45, unit 1127: native deletion.\nExact local replay matches all 128 original intervention reader vectors.\nAlgebraic numerator/RMS attribution sums to the total. Raw effects.\n16 articles. Descriptive 95% article-bootstrap intervals.',fontsize=9)
    fig.savefig('figures/native_unit45_normalization.png',dpi=180);fig.savefig('figures/native_unit45_normalization.pdf');plt.close(fig)


if __name__=='__main__':main()
