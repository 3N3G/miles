"""Extract post hoc illustrations from completed native expert experiments.

No model inference or new hypothesis test. Selection rules accompany every
example, and copied measurements retain their source-file hashes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('/home/ubuntu/inkling-interp-data'))
    p.add_argument('--tokenizer',type=Path,default=Path('/tmp/inkling-operator-tokenizer/tokenizer.json'))
    p.add_argument('--out',type=Path,default=Path('evidence/expert_blog_examples.json'));a=p.parse_args()
    root=a.data/'native-unit-atlas-confirmation-v1';tok=Tokenizer.from_file(str(a.tokenizer));sources={}
    def read(path):
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    inputs=read(root/'frozen_inputs.json')['records'];results=read(root/'results.json')
    profiles=read(root/'unit_activity_profiles.json');analysis=read(root/'analysis.json')
    def factors(q,expert,unit):
        rows=read(root/(q['id']+'-units.json'));z={}
        for r in rows:
            if r['expert']!=expert or r['unit']!=unit:continue
            z[r['visit']]={pos:{'raw_activation':h,'weighted_activation':w,'raw_gate':g,'raw_up':u}
                          for pos,h,w,g,u in zip(r['positions'],r['raw_activation'],r['weighted_activation'],r['raw_gate'],r['raw_up']) if pos>=16}
        return z
    def token_case(q,z,pos):
        assert pos in z[1] and pos in z[3]
        return {'position':pos,'token_id':q['token_ids'][pos],'token':tok.decode([q['token_ids'][pos]]),
                'context':tok.decode(q['token_ids'][max(0,pos-3):pos+4]),'pass1':z[1][pos],'pass3':z[3][pos]}
    def reader(q,expert,unit):
        rr=[r for r in results if r['id']==q['id'] and r['expert']==expert and r['unit']==unit]
        return [{k:r[k] for k in ['visit','native_block2_weight','actual_delta','predicted_full','predicted_dot_only','random_deltas']} for r in sorted(rr,key=lambda r:r['visit'])]
    first=read(a.data/'native-unit-routing-v1/representative_example.json')
    zfirst={v:{'raw_activation':first['unit_values'][v]['raw_activation'],
               'weighted_activation':first['unit_values'][v]['weighted_activation'],
               'raw_gate':first['unit_values'][v]['raw_gate'],'raw_up':first['unit_values'][v]['raw_up']} for v in ['1','3']}
    original={'article':first['article'],'input_id':first['id'],'expert':45,'unit':1127,
              'selection':first['selection'],'token':first['token'],'context':first['context'],
              'pass1':zfirst['1'],'pass3':zfirst['3'],'reader_effects':first['reader_effects']}
    q=next(q for q in inputs if q['id']=='unit-atlas-test-012');z=factors(q,52,585)
    eligible=[pos for pos in set(z.get(1,{}))&set(z.get(3,{})) if z[1][pos]['weighted_activation']<0<z[3][pos]['weighted_activation'] and tok.decode([q['token_ids'][pos]]).strip().isalpha()]
    changes=np.array([abs(z[3][pos]['weighted_activation']-z[1][pos]['weighted_activation']) for pos in eligible]);median=np.median(changes)
    pos=min(eligible,key=lambda pos:(abs(abs(z[3][pos]['weighted_activation']-z[1][pos]['weighted_activation'])-median),pos))
    gate={'article':q['article'],'input_id':q['id'],'expert':52,'unit':585,
          'selection':'Previously fixed median-gain article. Token nearest the median activity change among alphabetic negative-to-positive pieces selected on both passes.',
          'example':token_case(q,z,pos),'reader':reader(q,52,585)}
    candidates=[]
    for q in inputs:
        z=factors(q,83,1107)
        eligible=[pos for pos in set(z.get(1,{}))&set(z.get(3,{})) if z[1][pos]['weighted_activation']>0>z[3][pos]['weighted_activation'] and tok.decode([q['token_ids'][pos]]).strip().isalpha()]
        if not eligible:continue
        rr=reader(q,83,1107);candidates.append((q,z,eligible,rr,rr[1]['actual_delta']-rr[0]['actual_delta']))
    median=np.median([r[-1] for r in candidates]);q,z,eligible,rr,_=min(candidates,key=lambda r:(abs(r[-1]-median),r[0]['id']))
    changes=[abs(z[3][pos]['weighted_activation']-z[1][pos]['weighted_activation']) for pos in eligible];median=np.median(changes)
    pos=min(eligible,key=lambda pos:(abs(abs(z[3][pos]['weighted_activation']-z[1][pos]['weighted_activation'])-median),pos))
    opposite={'article':q['article'],'input_id':q['id'],'expert':83,'unit':1107,
              'selection':'Article nearest the median pass3-minus-pass1 raw reader effect among articles with an alphabetic positive-to-negative paired piece. Token nearest median activity change within eligible pieces.',
              'example':token_case(q,z,pos),'reader':rr}
    q=next(q for q in inputs if q['id']=='unit-atlas-test-030');z=factors(q,44,492)
    assert z[1][17]['weighted_activation']<0<z[3][17]['weighted_activation']
    assert z[1][51]['weighted_activation']>0>z[3][51]['weighted_activation']
    windows=[]
    for start,end,highlight in [(16,24,17),(47,55,51)]:
        positions=list(range(start,end));windows.append({'positions':positions,'highlight_position':highlight,
            'tokens':[tok.decode([q['token_ids'][pos]]) for pos in positions],
            'raw_activity':{str(v):[z.get(v,{}).get(pos,{}).get('raw_activation') for pos in positions] for v in [1,3]},
            'weighted_activity':{str(v):[z.get(v,{}).get(pos,{}).get('weighted_activation') for pos in positions] for v in [1,3]}})
    mixed={'article':q['article'],'input_id':q['id'],'expert':44,'unit':492,
           'selection':'Curated counterexample to a uniform token-level clock. Two readable positions in one paragraph with opposite activity-sign changes. No population claim is inferred from this curation.',
           'examples':[token_case(q,z,pos) for pos in [17,51]],'windows':windows,'reader':reader(q,44,492)}
    replay=read(Path('evidence/native_unit_replay_result.json'));text=read(Path('evidence/native_expert_text_example.json'))
    math=read(Path('evidence/native_expert_confirmation_example.json'));routing=read(Path('evidence/native_expert_screen_result.json'))['prospective_routing_105']
    math_results=read(Path('evidence/native_expert_confirmation_result.json'))
    q=next(q for q in read(Path('evidence/native_expert_text.json'))['records'] if q['id']=='text-confirmation-025')
    route_path=a.data/'expert-screen-v3/rank0/text-confirmation-025.npz';sources[str(route_path)]=hashlib.sha256(route_path.read_bytes()).hexdigest()
    with np.load(route_path) as data:
        assert np.array_equal(data['token_ids'].reshape(-1),q['token_ids'])
        selected=[bool((data['indices'][v,88]==105).any()) for v in range(3)]
    rr=read(a.data/'expert-screen-v3/rank0/text_results.json');rr={r['condition']:r for r in rr if r['id']==q['id']}
    i=rr['clean']['prediction_positions'].index(88);native=rr['clean']['nll'][i];deleted=rr['105-all']['nll'][i]
    assert tok.decode([q['token_ids'][88]])=='196' and tok.decode([q['token_ids'][89]])=='9'
    name_case={'article':q['article'],'input_id':q['id'],'selection':'Curated numerical string inside a musician stage name, from the existing text example.',
               'context':'DJ Product © 1969','input_position':88,'input_piece':'196','expert105_selected_per_pass':selected,
               'next_target_position':89,'next_target_piece':'9','native_next_target_nll':native,'deleted_next_target_nll':deleted,
               'loss_change':deleted-native,'native_probability':float(np.exp(-native)),'deleted_probability':float(np.exp(-deleted)),
               'intervention':'Expert105 removed at all routed positions on all three passes. The effect is not isolated to the displayed token.'}
    result={'scope':'Post hoc illustrations from existing experiments. No new model inference, fitting or statistical test.',
            'physical_layer':12,'expert_and_unit_ids':'zero based','original_normalization_case':original,
            'gate_case':gate,'opposite_activity_case':opposite,'mixed_activity_case':mixed,
            'normalization_decomposition':[r for r in replay['effects'] if r['condition']=='unit45'],
            'activity_profiles':profiles,'fixed_test':analysis,'number_routing':routing,'number_text_example':text,
            'number_name_case':name_case,'math_example':math,
            'math_test':{k:math_results[k] for k in ['primary','unique_questions','orders_per_question']},'source_sha256':sources,
            'reading_sources':[
                {'title':'Interpreting Inkling 2 Mini','url':'https://www.goodfire.com/research/interpreting-inkling-2-mini/','role':'Prior block-routing and outlier results.'},
                {'title':'Under the Hood of a Reasoning Model','url':'https://www.goodfire.com/research/under-the-hood-of-a-reasoning-model','role':'Example-driven exposition, with observation, intervention and bounded interpretation.'},
                {'title':'Understanding and Steering Llama 3 with Sparse Autoencoders','url':'https://www.goodfire.com/research/understanding-and-steering-llama-3','role':'Structure connecting feature observations with causal interventions.'}]}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'gate_case':gate,'opposite_case':opposite,'mixed_case':{k:v for k,v in mixed.items() if k!='windows'}},indent=2))


if __name__=='__main__':main()
