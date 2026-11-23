"""Reproduce old-fit paired unit sign counts without new model inference."""

import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--plot',action='store_true');a=p.parse_args()
    root=a.run/'native_neurons';records=json.loads((root/'inputs.json').read_text())
    targets=[(45,1127),(3,664),(45,130),(3,269),(105,156),(105,811)]
    z={k:np.zeros((144,64,3)) for k in targets};hits={k:np.zeros((144,64,3),dtype=bool) for k in targets}
    for doc in range(144):
        f=np.load(root/f'{doc:03}.npz');j=0
        while f'{j}_expert' in f:
            e=int(f[f'{j}_expert']);v=int(f[f'{j}_visit'])-1;positions=f[f'{j}_positions']
            for key in targets:
                if e==key[0]:
                    z[key][doc,positions,v]=f[f'{j}_activation'][:,key[1]]*f[f'{j}_weights'];hits[key][doc,positions,v]=True
            j+=1
    result=[];arrays={}
    for key in targets:
        for split,docs in [('fit',slice(0,128)),('validation',slice(128,144))]:
            values=z[key][docs,16:];hit=hits[key][docs,16:];both=hit[:,:,0]&hit[:,:,2]
            early=values[:,:,0][both];late=values[:,:,2][both]
            arrays[f'{key[0]}_{key[1]}_{split}']=np.stack([early,late],-1)
            result.append({'expert':key[0],'unit':key[1],'split':split,'paragraphs':values.shape[0],
                'paired_selected_positions':len(early),'negative_to_positive':int(((early<0)&(late>0)).sum()),
                'positive_to_negative':int(((early>0)&(late<0)).sum()),'paired_weighted_means':[float(early.mean()),float(late.mean())],
                'per_pass':[{'visit':v+1,'selected_positions':int(hit[:,:,v].sum()),
                    'mean_selected_weighted_activation':float(values[:,:,v][hit[:,:,v]].mean()),
                    'mean_all_positions_weighted_activation':float(values[:,:,v].mean())} for v in range(3)]})
    (a.run/'phase_unit_profiles.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(a.run/'phase_unit_pairs.npz',**arrays)
    print(json.dumps([r for r in result if r['expert']==45 and r['unit']==1127],indent=2))
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        data=arrays['45_1127_fit'];r=next(r for r in result if r['expert']==45 and r['unit']==1127 and r['split']=='fit')
        fig,ax=plt.subplots(figsize=(6.4,5.2));ax.scatter(data[:,0],data[:,1],s=9,alpha=.35,color='#0072B2',rasterized=True)
        ax.axvline(0,color='#777777',lw=.8);ax.axhline(0,color='#777777',lw=.8)
        ax.set(xlabel='Pass 1: routing-weighted unit activation',ylabel='Pass 3: routing-weighted unit activation',
               title='A native unit changes sign across loop passes')
        ax.text(.03,.97,f"{r['negative_to_positive']}/{r['paired_selected_positions']} positions: negative → positive\n{r['positive_to_negative']}/{r['paired_selected_positions']}: positive → negative",transform=ax.transAxes,va='top',fontsize=10)
        fig.subplots_adjust(bottom=.23)
        fig.text(.12,.035,'Inkling 2 Mini: layer 12, expert 45, unit 1127. Each point is one token.\nDescriptive fitting data: 128 paragraphs from 106 articles. No independence\nassumption across tokens. The causal routing test is separate.',fontsize=9)
        Path('figures').mkdir(exist_ok=True)
        fig.savefig('figures/native_unit45_phase_sign.png',dpi=180);fig.savefig('figures/native_unit45_phase_sign.pdf');plt.close(fig)


if __name__=='__main__':main()
