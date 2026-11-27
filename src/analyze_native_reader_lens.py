"""Exact-match scoring of the frozen native reader comparison, without a judge."""
import argparse
import json
import re
from pathlib import Path
import numpy as np


def canonical(s):return s.strip().casefold()


def named(text,label):
    return bool(re.search(r'(?<!\w)'+re.escape(label.strip())+r'(?!\w)',text,re.I))


def summarize(rows):
    return {'n':len(rows),**{m:sum(r['hit'][m] for r in rows) for m in ['partial_buffer','native_attention_reader']},
            'native_only':sum(r['hit']['native_attention_reader'] and not r['hit']['partial_buffer'] for r in rows),
            'buffer_only':sum(r['hit']['partial_buffer'] and not r['hit']['native_attention_reader'] for r in rows)}


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    assert json.loads((a.run/'manifest.json').read_text())['status']=='completed'
    inputs=json.loads((a.run/'frozen_inputs.json').read_text());observed={r['id']:r for r in json.loads((a.run/'results.json').read_text())}
    assert len(observed)==106
    rows=[]
    for record in inputs['records']:
        row=observed[record['id']];labels={canonical(s) for s in record['labels']}
        hit={m:any(canonical(t['token']) in labels for read in rs for t in read['tokens']) for m,rs in row['readouts'].items()}
        # Pinned generation_config.json defines eos_token_id=200006. A prefix
        # that reached its budget cannot establish failure or a hidden label.
        completed=bool(row['generation']['token_ids'] and row['generation']['token_ids'][-1]==200006)
        target_mentioned=named(row['generation']['text'],record['target'])
        capability=target_mentioned if completed else None
        unspoken=(not any(named(row['generation']['text'],s) for s in record['labels'])) if completed else None
        rows.append({**record,**row,'hit':hit,'capable':capability,'unspoken':unspoken,
                     'generation_completed':completed,'target_mentioned_in_prefix':target_mentioned})
    math=[r for r in rows if r['family'] in ['arithmetic','chain']];assert len(math)==64
    delta=np.array([int(r['hit']['native_attention_reader'])-int(r['hit']['partial_buffer']) for r in math])
    draw=np.random.default_rng(1729).integers(0,64,(10000,64));ci=np.quantile(delta[draw].mean(-1),[.025,.975])
    by_family={family:summarize([r for r in rows if r['family']==family]) for family in ['arithmetic','chain','direct','practice']}
    by_visit={m:{str(v):sum(any(canonical(t['token']) in {canonical(s) for s in r['labels']} for t in next(x for x in r['readouts'][m] if x['visit']==v)['tokens']) for r in math) for v in inputs['read_visits']} for m in inputs['methods']}
    examples=[]
    for label,a_method,b_method in [('first_native_only','native_attention_reader','partial_buffer'),('first_buffer_only','partial_buffer','native_attention_reader')]:
        selected=next((r for r in math if r['hit'][a_method] and not r['hit'][b_method]),None)
        if selected:examples.append({'selection':label,**selected})
    if not examples:examples=[{'selection':'first_frozen_question',**rows[0]}]
    result={'primary_math':{**summarize(math),'native_minus_buffer_rate':float(delta.mean()),'paired_question_bootstrap95':ci.tolist(),
                            'resamples':10000,'seed':1729},'by_family':by_family,'math_by_visit':by_visit,
            'secondary_capable_unspoken_math':summarize([r for r in math if r['capable'] is True and r['unspoken'] is True]),
            'capability_counts':{family:{'completed_correct':sum(r['capable'] is True for r in rows if r['family']==family),
                'completed_incorrect':sum(r['capable'] is False for r in rows if r['family']==family),
                'truncated_unknown':sum(r['capable'] is None for r in rows if r['family']==family),
                'n':sum(r['family']==family for r in rows)} for family in by_family},
            'examples':examples,'limits':['Exact token matching misses paraphrases and alternative concept spellings.',
                'The arithmetic intermediate is a benchmark hypothesis, not a proven uniquely necessary computation.',
                'Direct and public practice tasks test answer visibility rather than hidden intermediate computations.',
                'The comparison is fixed and unfitted but only covers these prompts and ten chosen visits.',
                'Capability requires a completed generation, with a conservative word/number match. Truncated twelve-token prefixes remain unknown.',
                'Without complete outputs, the test cannot establish that intermediates stay unspoken.']}
    (a.run/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');(a.run/'scored_questions.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['primary_math','by_family','capability_counts','secondary_capable_unspoken_math']},indent=2))


if __name__=='__main__':main()
