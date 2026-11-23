"""Make figures for five native-unit case studies from saved measurements."""

import argparse,json
from pathlib import Path
import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    p=argparse.ArgumentParser();p.add_argument('--examples',type=Path,default=Path('evidence/expert_blog_examples.json'))
    p.add_argument('--out',type=Path,default=Path('figures/expert_cases'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    r=json.loads(a.examples.read_text());blue='#0072B2';orange='#D55E00'
    def save(fig,name):
        fig.savefig(a.out/(name+'.png'),dpi=180);fig.savefig(a.out/(name+'.pdf'));plt.close(fig)
    first=r['original_normalization_case'];second=r['gate_case']['example']
    fig,axes=plt.subplots(1,2,figsize=(10,5.2),sharey=True)
    for ax,case,title in zip(axes,[first,second],['Expert 45, unit 1127\nC in While K.C.','Expert 52, unit 585\nFormation in Wadhurst Clay Formation']):
        x=np.arange(3)
        for visit,offset,color in [('pass1',-.09,blue),('pass3',.09,orange)]:
            vals=[case[visit][k] for k in ['raw_gate','raw_up','raw_activation']]
            ax.scatter(x+offset,vals,color=color,label='Pass 1' if visit=='pass1' else 'Pass 3',s=42,zorder=3)
            for xx,yy in zip(x+offset,vals):
                ax.annotate(f'{yy:+.2f}',(xx,yy),xytext=(-5 if visit=='pass1' else 5,8 if yy>=0 else -14),textcoords='offset points',ha='right' if visit=='pass1' else 'left',fontsize=9)
        ax.axhline(0,lw=.9,color='#777777');ax.set(xticks=x,xticklabels=['Gate input','Up input','Unit output'],title=title,xlim=(-.45,2.45),ylim=(-2.55,2))
    axes[0].set_ylabel('Native scalar value');axes[1].legend(frameon=False,loc='upper right')
    fig.suptitle('Two units reverse sign through different input factors',fontsize=14,y=.98)
    fig.subplots_adjust(left=.09,right=.98,top=.78,bottom=.24,wspace=.15)
    fig.text(.09,.045,'Inkling 2 Mini, physical layer 12. Two illustrative token positions, not population averages.\nThe gate stays negative in unit 45:1127. The up input stays negative in unit 52:585.\nDisplayed unit outputs precede the positive expert routing weight.',fontsize=9)
    save(fig,'gate_and_up')
    fig,axes=plt.subplots(1,2,figsize=(10,5.2));colors=['#009E73','#CC79A7']
    for expert,unit,color in [(52,585,colors[0]),(83,1107,colors[1])]:
        profile=next(x for x in r['activity_profiles']['units'] if x['expert']==expert and x['unit']==unit)
        axes[0].plot([0,1],profile['paired_mean_early_late_h'],'o-',color=color,label=f'Expert {expert}, unit {unit}')
        effect=next(x for x in r['fixed_test']['per_unit'] if x['expert']==expert and x['unit']==unit)
        means=np.array([x['mean'] for x in effect['adjusted_effects']])*100;bounds=np.array([x['interval'] for x in effect['adjusted_effects']])*100
        axes[1].errorbar([0,1],means,yerr=np.stack([means-bounds[:,0],bounds[:,1]-means]),fmt='o-',color=color,capsize=4)
    for ax in axes:ax.axhline(0,lw=.9,color='#777777');ax.set(xticks=[0,1],xticklabels=['Pass 1','Pass 3'],xlim=(-.3,1.3))
    axes[0].set(title='Opposite activity changes',ylabel='Mean routed unit activity')
    axes[1].set(title='Matching routing effects',ylabel='Deletion minus random (percentage points)')
    axes[0].legend(frameon=False,fontsize=9)
    fig.subplots_adjust(left=.09,right=.98,top=.85,bottom=.27,wspace=.36)
    fig.text(.09,.045,'Layer 12 units → layer 13 Block 2 read weight. Positive means removal increases the weight.\nLeft: descriptive means over 70 paired positions for unit 52 and 200 for unit 83.\nRight: 32 articles, three norm-matched random controls. Intervals adjust for eight tests.\nDifferent units write different vectors. Their scalar signs do not define a shared routing meaning.',fontsize=9)
    save(fig,'activity_and_routing')
    case=r['mixed_activity_case'];fig,axes=plt.subplots(2,1,figsize=(10,5.5))
    arrays=[np.array([w['raw_activity']['1'],w['raw_activity']['3']],dtype=float) for w in case['windows']]
    limit=max(float(np.nanmax(abs(x))) for x in arrays)
    cmap=LinearSegmentedColormap.from_list('activity',[blue,'#ffffff',orange]);cmap.set_bad('#dddddd')
    for ax,w,values in zip(axes,case['windows'],arrays):
        ax.imshow(np.ma.masked_invalid(values),cmap=cmap,vmin=-limit,vmax=limit,aspect='auto')
        labels=[t.strip() or 'space' for t in w['tokens']]
        ax.set(xticks=np.arange(len(labels)),xticklabels=labels,yticks=[0,1],yticklabels=['Pass 1','Pass 3'])
        for i in range(2):
            for j in range(values.shape[1]):
                if np.isfinite(values[i,j]):
                    label='≈0' if abs(values[i,j])<.005 else f'{values[i,j]:+.2f}'
                    ax.text(j,i,label,ha='center',va='center',fontsize=9,color='white' if abs(values[i,j])>limit*.6 else '#222222')
                else:ax.text(j,i,'—',ha='center',va='center',color='#777777')
        j=w['positions'].index(w['highlight_position']);ax.add_patch(Rectangle((j-.5,-.5),1,2,fill=False,edgecolor='#111111',lw=1.8))
        ax.set_title('Word number' if w['highlight_position']==17 else 'Comma after Conlon',loc='left',fontsize=11)
    fig.suptitle('One unit changes sign in opposite directions within one paragraph',fontsize=14,y=.98)
    fig.subplots_adjust(left=.09,right=.98,top=.84,bottom=.24,hspace=.70)
    fig.text(.09,.045,'Layer 12, expert 44, unit 492. Two excerpts from the same Stuart McCall paragraph.\nCells show rounded unit output. Blue is negative, orange is positive, and gray means not routed.\nBoxes mark the two illustrated sign reversals. These are clean activations, not token-specific causal effects.',fontsize=9)
    save(fig,'mixed_token_responses')
    fig,axes=plt.subplots(1,2,figsize=(10,4.8));rr=r['number_routing']
    means=np.array([x['rate_ratio'] for x in rr]);bounds=np.array([x['family95_interval'] for x in rr]);y=np.arange(3)
    axes[0].errorbar(means,y,xerr=np.stack([means-bounds[:,0],bounds[:,1]-means]),fmt='o',color=blue,capsize=4)
    axes[0].axvline(1,lw=.9,color='#777777');axes[0].set(yticks=y,yticklabels=['Pass 1','Pass 2','Pass 3'],xlabel='Cue routing rate / other routing rate',xlim=(0,5.3),title='Expert 105 prefers number and time cues');axes[0].invert_yaxis()
    loss=json.loads(Path('evidence/native_expert_screen_result.json').read_text())
    loss=next(x for x in loss['prospective_text'] if x['condition']=='105-all')['contrasts']
    keys=['numeric_target','numeric_input'];mean=np.array([loss[k]['mean'] for k in keys]);bounds=np.array([loss[k]['interval'] for k in keys])
    axes[1].errorbar(mean,[0,1],xerr=np.stack([mean-bounds[:,0],bounds[:,1]-mean]),fmt='o',color=orange,capsize=4)
    axes[1].axvline(0,lw=.9,color='#777777');axes[1].set(yticks=[0,1],yticklabels=['Cue target','Cue input'],xlabel='Extra deletion loss on cues (nats)',title='Primary necessity contrasts include zero',ylim=(-.6,1.6));axes[1].invert_yaxis()
    fig.subplots_adjust(left=.09,right=.98,top=.82,bottom=.25,wspace=.40)
    fig.text(.09,.045,'Physical layer 12. Native expert removal on all three passes.\nLeft: 48 articles, intervals adjust for three tests. Right: 42 eligible articles, two-test correction.\nCues include digits, number words, ordinals and time-unit words. Uncertainty resamples articles.',fontsize=9)
    save(fig,'number_routing_and_loss')


if __name__=='__main__':main()
