"""Describe exact native reader replay without adding a primary significance test."""

import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    rows=json.loads((a.run/'results.json').read_text())
    assert len(rows)==128
    rng=np.random.default_rng(1729);draw=rng.integers(0,16,(10000,16))
    effects=[]
    for name in ['unit45','unit3','joint','control45']:
        for visit in [1,3]:
            selected=sorted([r for r in rows if r['condition']==name and r['visit']==visit],key=lambda r:r['id'])
            assert len(selected)==16
            record={'condition':name,'visit':visit,'articles':16}
            for key in ['delta_block2_weight','numerator_contribution','rms_contribution']:
                values=np.array([np.mean(r[key]) for r in selected])
                record[key]={'mean':float(values.mean()),'descriptive_95_interval':np.quantile(values[draw].mean(-1),[.025,.975]).tolist()}
            effects.append(record)
    source=json.loads((a.run/'unit_projection_sources.json').read_text());sources=[];paired=[]
    fields=['native','current_depth_path','direct_previous_loop_path','normalization_cast_residual','projection_rounding_residual']
    for label in ['gate','up']:
        for visit in [1,2,3]:
            selected=[r for r in source if r['projection']==label and r['visit']==visit]
            sources.append({'projection':label,'visit':visit,'selected_positions':sum(len(r['positions']) for r in selected),
                            'token_weighted_means':{k:float(np.mean([v for r in selected for v in r[k]])) for k in fields}})
    for article in sorted({r['id'] for r in source}):
        selected={(r['projection'],r['visit']):r for r in source if r['id']==article}
        first=selected['up',1];third=selected['up',3]
        index1={v:i for i,v in enumerate(first['positions'])};index3={v:i for i,v in enumerate(third['positions'])}
        for position in sorted(set(index1)&set(index3)):
            i=index1[position];j=index3[position]
            paired.append({'id':article,'position':position,'up_pass1':first['native'][i],'up_pass3':third['native'][j],
                           'up_current_path_pass3':third['current_depth_path'][j],
                           'up_message_path_pass3':third['direct_previous_loop_path'][j]})
    flips=[r for r in paired if r['up_pass1']>0 and r['up_pass3']<0]
    result={'effects':effects,'projection_sources':sources,
            'paired_positions':len(paired),'positive_to_negative_up':len(flips),
            'current_path_already_negative_among_flips':sum(r['up_current_path_pass3']<0 for r in flips),
            'maximum_replayed_reader_weight_error':max(r['maximum_full_model_prediction_error'] for r in rows),
            'maximum_attribution_sum_error':max(r['maximum_decomposition_sum_error'] for r in rows),
            'limits':['Exploratory explanation of the existing 16-article test, not another independent sample.',
                      'Attribution factors are algebraic and need not be separately realizable interventions.',
                      'The current depth path can already contain information from previous loops.',
                      'Projection means weight selected token positions, not independent articles.']}
    (a.run/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
