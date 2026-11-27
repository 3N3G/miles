"""Fixed comparison figures for the native reader token lens."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();d=json.loads((a.run/'analysis.json').read_text())
    fig,ax=plt.subplots(figsize=(8,4.9));x=np.arange(4);families=['arithmetic','chain','direct','practice']
    for method,offset,color,label in [('partial_buffer',-.19,'#999999','Partial block buffer'),('native_attention_reader',.19,'#0072B2','Native current reader')]:
        y=[d['by_family'][f][method]/d['by_family'][f]['n']*100 for f in families]
        ax.bar(x+offset,y,width=.36,color=color,label=label)
        for i,yy,f in zip(x+offset,y,families):ax.text(i,yy+1,f"{d['by_family'][f][method]}/{d['by_family'][f]['n']}",ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=['Two-operation\narithmetic','Sequential\narithmetic','Direct source\nlabels','Public\npractice'],ylim=(0,110),ylabel='Exact source-label recovery (%)',title='The native reader gives no exact-label recovery gain')
    ax.legend(frameon=False,loc='lower right');fig.subplots_adjust(bottom=.26)
    fig.text(.13,.02,'Same ten visits, last prompt token, same output head, no fitted weights.\nReadouts drop only control tokens and case/spacing duplicates.\nSix direct questions have model-dependent preference labels. Capability is not yet gated.\nThese counts do not establish recovery of hidden computations.',fontsize=9)
    for suffix in ['png','pdf']:fig.savefig('figures/native_reader_lens_recovery.'+suffix,dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.7))
    for method,color,label in [('partial_buffer','#999999','Partial block buffer'),('native_attention_reader','#0072B2','Native current reader')]:
        values=d['math_by_visit'][method];ax.plot([int(v) for v in values],[n/64*100 for n in values.values()],'-o',color=color,label=label)
    ax.set(xlabel='Effective visit (zero based)',ylabel='Arithmetic intermediate recovery (%)',ylim=(0,100),title='Arithmetic labels rarely surface before generation');ax.legend(frameon=False)
    fig.subplots_adjust(bottom=.25);fig.text(.13,.035,'64 new benchmark prompts. Each point is a separate single-visit readout.\nA line joins fixed sampled visits only. No smoothing or selection after evaluation.\nOverall recovery counts a concept seen at any of the ten visits.',fontsize=9)
    for suffix in ['png','pdf']:fig.savefig('figures/native_reader_lens_visits.'+suffix,dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
