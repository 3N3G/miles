"""Freeze unused benchmark questions for an unfitted native-reader comparison."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

ROOT=Path('/home/ubuntu/inkling-interp-data/workspace-source')
TOKENIZER=Path('/tmp/inkling-operator-tokenizer/tokenizer.json')


def chat(prompt):
    return '<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>'+prompt+'<|end_message|><|message_model|>'


def main():
    dest=Path('evidence/native_reader_lens_inputs.json');assert not dest.exists()
    tok=Tokenizer.from_file(str(TOKENIZER));rng=np.random.default_rng(161803)
    practice_path=Path('/home/ubuntu/inkling-interp-data/challenge-practice/practice/private/questions.jsonl')
    practice=[json.loads(s) for s in practice_path.read_text().splitlines()]
    old_prompts={r['prompt'].strip() for f in ['evidence/pilot_items.json','evidence/transport_development.json','evidence/transport_final_items.json'] for r in json.loads(Path(f).read_text())['items']}
    old_prompts.update(r['prompt'].strip() for r in practice)
    records=[];sources=[];availability={}
    for family,file in [('arithmetic','evals/computational/arithmetic_intermediates/items_v2.json'),
                        ('chain','evals/computational/chain_intermediates/items_v2.json'),
                        ('direct','evals/basic/basic_readout_st/items_v8.json')]:
        path=ROOT/file;data=json.loads(path.read_text());items=data['items'] if isinstance(data,dict) else data
        candidates=[]
        for item in items:
            if item['prompt'].strip() in old_prompts:continue
            labels=[str(x) for x in item['intermediates']]
            # A one-token lens cannot emit a whole multi-token label. Keep the
            # first source-defined intermediate only, with reachable forms.
            label=labels[0]
            forms=[s for s in [label,label.title(),label.upper()] if any(len(tok.encode(pre+s,add_special_tokens=False).ids)==1 for pre in ['', ' '])]
            if not forms:continue
            if re.search(r'(?<!\w)'+re.escape(label)+r'(?!\w)',item['prompt'],re.I):continue
            model_input=chat(item['prompt']);ids=tok.encode(model_input,add_special_tokens=False).ids
            candidates.append({'id':family+'-'+item['name'],'family':family,'prompt':item['prompt'],'model_input':model_input,
                               'token_ids':ids,'read_position':len(ids)-1,'read_token':tok.decode([ids[-1]]),
                               'labels':[label],'target':str(item.get('target',item.get('answer'))),
                               'source_file':file,'source_name':item['name'],'all_source_intermediates':labels})
        availability[family]=len(candidates);rng.shuffle(candidates);assert len(candidates)>=32,(family,len(candidates))
        records.extend(candidates[:32]);sources.append({'file':file,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for item in practice:
        ids=tok.encode(item['model_input'],add_special_tokens=False).ids
        assert len(ids)-1==item['read_position']
        records.append({'id':item['id'],'family':'practice','prompt':item['prompt'],'model_input':item['model_input'],
                        'token_ids':ids,'read_position':item['read_position'],'read_token':item['read_token'],
                        'labels':[str(item['intermediate'])],'target':str(item['answer'])})
    value={'source_commit':'abd152e20aa2fa5d7df7f01ade6759ebe0f858fb','sources':sources,
           'practice_sha256':hashlib.sha256(practice_path.read_bytes()).hexdigest(),'selection_seed':161803,
           'unused_eligible_counts':availability,'read_visits':[12,24,34,35,36,43,44,45,46,47],
           'methods':['partial_buffer','native_attention_reader'],'records':records}
    dest.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'eligible':availability,'count':len(records),'examples':[{k:r[k] for k in ['family','model_input','labels','target']} for r in [records[0],records[32],records[64],records[96]]]},indent=2))


if __name__=='__main__':main()
