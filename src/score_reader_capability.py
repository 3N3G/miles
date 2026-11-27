"""Merge repaired completions with fixed readouts, preserving the primary score."""
import argparse
import json
import re
from fractions import Fraction
from pathlib import Path
from analyze_native_reader_lens import named,summarize


def numeric_correct(text,target):
    # Require exactly one final-channel numeric expression. Do not grade numbers
    # mentioned during thinking as answers. Unparseable formats stay unknown.
    text=re.sub(r'(?<=\d),(?=\d)','',text)
    numbers=re.findall(r'(?<![\w.])[+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?(?![\w.])',text.rstrip('.'))
    if len(numbers)!=1:return None
    try:return abs(float(Fraction(numbers[0].replace(' ',''))-Fraction(target)))<=1e-6
    except (ValueError,ZeroDivisionError):return None


def main():
    p=argparse.ArgumentParser();p.add_argument('lens_run',type=Path);p.add_argument('completion_run',type=Path);a=p.parse_args()
    assert json.loads((a.completion_run/'manifest.json').read_text())['status']=='completed'
    original=json.loads((a.lens_run/'analysis.json').read_text());rows=json.loads((a.lens_run/'scored_questions.json').read_text())
    completed={r['id']:r for r in json.loads((a.completion_run/'results.json').read_text())};assert len(completed)==74
    for r in rows:
        if r['id'] not in completed:continue
        out=completed[r['id']];r['repaired_generation']=out
        correct=None
        if out['completed']:
            if r['family'] in ['arithmetic','chain']:correct=numeric_correct(out['final_text'],r['target'])
            else:correct=named(out['final_text'],r['target']) or (r['target']=='7' and named(out['final_text'],'seven'))
        r['capable']=correct
        text=re.sub(r'(?<=\d),(?=\d)','',out['text'])
        r['unspoken']=not any(named(text,s) for s in r['labels']) if out['completed'] else None
        if r['family']=='practice' and r['labels']==['7'] and named(text,'seven'):r['unspoken']=False
    math=[r for r in rows if r['family'] in ['arithmetic','chain']]
    assert summarize(math)=={k:original['primary_math'][k] for k in ['n','partial_buffer','native_attention_reader','native_only','buffer_only']}
    capability={}
    for family in ['arithmetic','chain','direct','practice']:
        family_rows=[r for r in rows if r['family']==family]
        capability[family]={'n':len(family_rows),'greedy_correct':sum(r['capable'] is True for r in family_rows),
                            'greedy_incorrect':sum(r['capable'] is False for r in family_rows),'unknown':sum(r['capable'] is None for r in family_rows)}
    result={'primary_readout_score_unchanged':True,'capability':capability,
            'math_greedy_correct':summarize([r for r in math if r['capable'] is True]),
            'math_greedy_correct_and_unspoken':summarize([r for r in math if r['capable'] is True and r['unspoken'] is True]),
            'math_greedy_correct_but_spoken':summarize([r for r in math if r['capable'] is True and r['unspoken'] is False]),
            'raw_example':next((r for r in math if r['capable'] is True),math[0]),
            'limits':['This checks one deterministic greedy completion, not the original benchmark stochastic gate.',
                      'The unspoken check counts intermediate mentions in native thinking text too.',
                      'Unparseable final answers and truncated completions remain unknown.',
                      'The 32 direct-control completions retain their original short budgets.']}
    (a.completion_run/'capability_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.completion_run/'merged_scored_questions.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['raw_example']},indent=2))


if __name__=='__main__':main()
