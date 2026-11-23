"""Describe native units and freeze an old-fit-only numeric routing family."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from inspect_native_experts import numeric_cue


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--tokenizer',type=Path,required=True);a=p.parse_args()
    root=a.run/'native_neurons';records=json.loads((root/'inputs.json').read_text());assert len(records)==144
    assert sum(r['split']=='fit' for r in records)==128
    tok=Tokenizer.from_file(str(a.tokenizer));norms=np.load(root/'decoder_column_norms.npz')
    units=defaultdict(list);route_counts=defaultdict(lambda:np.zeros((128,4)))
    for doc,r in enumerate(records):
        cue=np.array([numeric_cue(tok.decode([i])) for i in r['token_ids']])
        arrays=np.load(root/r['file']);j=0
        while f'{j}_expert' in arrays:
            e=int(arrays[f'{j}_expert']);visit=int(arrays[f'{j}_visit']);positions=arrays[f'{j}_positions']
            mask=positions>=16;positions=positions[mask]
            h=arrays[f'{j}_activation'][mask];w=arrays[f'{j}_weights'][mask]
            units[e].append({'doc':doc,'visit':visit,'positions':positions,'h':h,'w':w,'cue':cue[positions],'split':r['split']})
            j+=1
        routes=np.load(root/f'core_routing_{doc:03}.npz')
        for k,(layer,visit) in enumerate(zip(routes['layer_indices'],routes['visits'])):
            ix=routes['indices'][k,16:];current=cue[16:]
            counts=route_counts[(r['split'],int(layer),int(visit))]
            counts[:,0]+=np.bincount(ix[current].reshape(-1),minlength=128)
            counts[:,1]+=current.sum()
            counts[:,2]+=np.bincount(ix[~current].reshape(-1),minlength=128)
            counts[:,3]+=(~current).sum()
    route_rows=[];family=[];controls=[]
    for layer in range(6,18):
        pooled=sum(route_counts[('fit',layer,v)] for v in [1,2,3])
        yes=(pooled[:,0]+.5)/(pooled[:,1]+1);no=(pooled[:,2]+.5)/(pooled[:,3]+1)
        total=pooled[:,0]+pooled[:,2];difference=yes-no
        eligible=np.where(total>=200)[0];expert=int(eligible[np.argmax(difference[eligible])])
        control_candidates=[e for e in eligible if e!=expert and yes[e]/no[e]<1.5]
        control=min(control_candidates,key=lambda e:(abs(total[e]-total[expert]),e))
        family.append({'layer_index':layer,'expert':expert,'fit_cue_rate_difference':float(difference[expert]),'fit_count':int(total[expert])})
        controls.append({'layer_index':layer,'expert':int(control),'fit_cue_rate_difference':float(difference[control]),'fit_count':int(total[control])})
        for split in ['fit','validation']:
            for v in [1,2,3]:
                for e in range(128):
                    c=route_counts[(split,layer,v)][e]
                    route_rows.append({'split':split,'layer_index':layer,'visit':v,'expert':e,
                                       'numeric_selected':int(c[0]),'numeric_total':int(c[1]),
                                       'other_selected':int(c[2]),'other_total':int(c[3])})
    selection={'selection_data':'128 old fit paragraphs only, no question performance or new article states',
               'family_rule':'One expert per core layer maximizing cue-rate difference, at least200 selections pooled over visits',
               'control_rule':'Closest pooled selection count among experts with smoothed numeric route enrichment below1.5',
               'numeric_family':family,'frequency_matched_family':controls,'unit':'Expert identity includes its physical layer index'}
    (a.run/'numeric_routing_family.json').write_text(json.dumps(selection,indent=2)+'\n')
    (a.run/'core_numeric_routing.json').write_text(json.dumps(route_rows,indent=2)+'\n')
    profiles=[];examples=[]
    preselected={105:[156,811],45:[1127,130],3:[664,269]}
    for expert,chunks in units.items():
        values=np.concatenate([c['h']*c['w'][:,None] for c in chunks]);cues=np.concatenate([c['cue'] for c in chunks])
        fit=np.concatenate([np.repeat(c['split']=='fit',len(c['positions'])) for c in chunks])
        ids=[(c['doc'],int(pos),c['visit']) for c in chunks for pos in c['positions']]
        train=values[fit];tc=cues[fit]
        contrast=(train[tc].mean(0)-train[~tc].mean(0))/np.maximum(train.std(0),1e-8)
        contribution=np.abs(train).mean(0)*norms[str(expert)]
        selected=sorted(set(preselected.get(expert,[])+np.argsort(contrast)[-2:].tolist()+[int(contribution.argmax())]))
        profiles.append({'expert':expert,'fit_selected_states':int(fit.sum()),'development_selected_states':int((~fit).sum()),
                         'fit_numeric_contrast':contrast.tolist(),'fit_mean_contribution_proxy':contribution.tolist(),'selected_units':selected})
        for unit in selected:
            for split,mask in [('fit',fit),('validation',~fit)]:
                positions=np.where(mask)[0];order=positions[np.argsort(np.abs(values[mask,unit]))[::-1][:4]]
                for idx in order:
                    doc,pos,visit=ids[idx];r=records[doc]
                    examples.append({'expert':expert,'unit':unit,'split':split,'id':r['id'],'visit':visit,'position':pos,
                                     'weighted_activation':float(values[idx,unit]),'numeric_cue':bool(cues[idx]),
                                     'token':tok.decode([r['token_ids'][pos]]),'context':tok.decode(r['token_ids'][max(0,pos-10):pos+1]),
                                     'full_input':r['rendered_input']})
    (a.run/'native_unit_profiles.json').write_text(json.dumps(profiles,indent=2)+'\n')
    (a.run/'native_unit_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    print(json.dumps(selection,indent=2));print('Native units selected from old fit activations. Examples',len(examples))


if __name__=='__main__':main()
