"""Standalone figures for the frozen extended-unit test."""
import argparse
import json
from pathlib import Path
import numpy as np


def line(ax,records,color,label):
    x=np.array([i+1 for i,r in enumerate(records) if r is not None]);records=[r for r in records if r is not None]
    y=np.array([r['mean'] for r in records]);ci=np.array([r['interval'] for r in records])
    ax.plot(x,y,color=color,label=label);ax.fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.16)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();d=json.loads((a.run/'analysis.json').read_text())
    fig,axes=plt.subplots(4,2,figsize=(10,10),sharex=True)
    for ax,u in zip(axes.flat,d['per_unit']):
        r=[{'mean':v['mean']*100,'interval':(np.array(v['interval'])*100).tolist()} for v in u['exact_effect_by_visit']]
        line(ax,r,'#0072B2',None);ax.axhline(0,color='#999999',lw=.7);ax.axvline(3.5,color='#999999',lw=.7,ls=':')
        ax.set_title(f"Expert {u['expert']}, unit {u['unit']}",fontsize=10);ax.set_xticks([1,3,6,9,12,15,18,21,24])
    for ax in axes[-1]:ax.set_xlabel('Pass number')
    fig.supylabel('Block 2 read-weight change on unit deletion (percentage points)')
    fig.suptitle('Native unit effects recur over extended passes')
    fig.tight_layout(rect=(.025,.065,1,.97));fig.text(.08,.018,'16 new WikiText articles, eight fixed layer-12 units. Read at native layer 13.\nShading: descriptive 95% article-bootstrap intervals. Extension begins after pass 3.\nFull native prefix and local replay checks are required before interpreting trajectories.',fontsize=9)
    for suffix in ['png','pdf']:fig.savefig('figures/extended_unit_effects.'+suffix,dpi=180)
    plt.close(fig)
    u=next(u for u in d['per_unit'] if (u['expert'],u['unit'])==(45,1127))
    r=[{'mean':v['mean']*100,'interval':(np.array(v['interval'])*100).tolist()} for v in u['exact_effect_by_visit']]
    fig,ax=plt.subplots(figsize=(8.4,4.5));line(ax,r,'#0072B2',None)
    ax.axhline(0,color='#555555',lw=.8);ax.axvline(3.5,color='#999999',lw=.9,ls=':')
    ax.set(xlabel='Pass number',ylabel='Change in Block 2 read weight\nwhen the unit is deleted (percentage points)',
           title='The original unit’s opposing routing effects recur',xticks=range(1,25),xlim=(.7,24.3))
    fig.subplots_adjust(left=.16,bottom=.26)
    fig.text(.16,.022,'Inkling 2 Mini, physical layer 12, expert 45, unit 1127. 16 new articles.\nPositive: deleting the unit makes the next layer read Block 2 more.\nNegative: deleting it makes the next layer read Block 2 less.\nShading: 95% article bootstrap. Dotted line: the trained three-pass horizon.',fontsize=9)
    for suffix in ['png','pdf']:fig.savefig('figures/extended_unit45_recurrence.'+suffix,dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.6,4.8))
    for lag,c in [(1,'#D55E00'),(3,'#0072B2'),(6,'#009E73')]:line(ax,d['state_distance_by_visit'][str(lag)],c,f'Lag {lag}')
    ax.set(xlabel='Pass number',ylabel='Normalized whole-state RMS distance',title='Three-pass state recurrence persists beyond training',ylim=(0,None))
    ax.legend(frameon=False);fig.subplots_adjust(bottom=.23);fig.text(.13,.035,'Both core blocks, all 128 token positions. 16 new articles.\nPer-token/block RMS normalization. Shading: 95% article-bootstrap intervals.\nPrimary lag-three versus lag-one test uses passes 19–24 with four-test correction.',fontsize=9)
    for suffix in ['png','pdf']:fig.savefig('figures/extended_state_recurrence.'+suffix,dpi=180)
    plt.close(fig)
    methods=['full','dot_only','zero'];means=np.array([d['mae_new_visits'][m]['mean'] for m in methods])*100
    ci=np.array([d['mae_new_visits'][m]['interval'] for m in methods])*100
    fig,ax=plt.subplots(figsize=(7.6,4.8));ax.bar(range(3),means,color=['#222222','#0072B2','#999999'])
    ax.errorbar(range(3),means,yerr=np.array([means-ci[:,0],ci[:,1]-means]),fmt='none',ecolor='#444444',capsize=4)
    ax.set(xticks=range(3),xticklabels=['With normalization','Dot product only','Predict zero'],ylabel='Mean absolute error (percentage points)',ylim=(0,None),title='Normalization lowers error on previously untested passes')
    fig.subplots_adjust(bottom=.23);fig.text(.13,.035,'Pass 2 and passes 4–24. Eight fixed units averaged within 16 articles.\nNo fitting or unit selection on these articles. Error bars: 95% article bootstrap.\nPrimary paired tests use a separate four-test correction.',fontsize=9)
    for suffix in ['png','pdf']:fig.savefig('figures/extended_forecast_error.'+suffix,dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
