import hashlib,json,random,runpy
from decimal import Decimal
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
root=Path('/home/ubuntu/.cache/miles/gsm8k-control')
rows=pq.read_table(root/'source/train.parquet').to_pylist()
print('columns',list(rows[0]),'rows',len(rows))
print('first',json.dumps(rows[0],ensure_ascii=False))
checker=runpy.run_path('/home/ubuntu/miles/miles/rollout/rm_hub/math_utils.py')['grade_answer_verl']
positive=negative=blank=0
for row in rows:
 label=row['label'];assert isinstance(label,str) and label
 assert checker(r'\boxed{'+label+'}',label)
 wrong=str(Decimal(label.replace(',',''))+1)
 assert not checker(r'\boxed{'+wrong+'}',label)
 assert not checker('',label)
 assert not checker('I do not know.',label)
 positive+=1;negative+=1;blank+=1
rng=random.Random(20261126);indices=list(range(len(rows)));rng.shuffle(indices);valid=set(indices[:256]);training=[(i,row) for i,row in enumerate(rows) if i not in valid];validation=[(i,row) for i,row in enumerate(rows) if i in valid]
assert len(training)==7217 and len(validation)==256
hashes={}
for name,group in [('train',training),('validation',validation)]:
 p=root/f'{name}.jsonl'
 with p.open('w') as f:
  for i,row in group:f.write(json.dumps({'source_row':i,**row},ensure_ascii=True)+'\n')
 hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
tok=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True)
first=training[0][1];formatted=tok.apply_chat_template(first['messages'],tokenize=False,add_generation_prompt=True)
(root/'formatted-example.txt').write_text(formatted)
example={'source_row':training[0][0],'messages':first['messages'],'label':first['label'],'formatted_prompt':formatted,'prompt_tokens':len(tok(formatted,add_special_tokens=False)['input_ids']),'correct_example':r'\boxed{'+first['label']+'}','correct_reward':int(checker(r'\boxed{'+first['label']+'}',first['label'])),'wrong_example':r'\boxed{'+str(Decimal(first['label'].replace(',',''))+1)+'}','wrong_reward':0}
(root/'example.json').write_text(json.dumps(example,indent=2)+'\n')
record={'dataset':{'repo_id':'zhuzilin/gsm8k','revision':'0cbd9f31d91ac21a7613dcbc7fef992adac459ae','file':'train.parquet','source_sha256':hashlib.sha256((root/'source/train.parquet').read_bytes()).hexdigest(),'rows':len(rows)},'split_seed':20261126,'training_rows':len(training),'validation_rows':len(validation),'official_test_reads':0,'partition_sha256':hashes,'grader_source':'miles/rollout/rm_hub/math_utils.py::grade_answer_verl','grader_source_sha256':hashlib.sha256(Path('/home/ubuntu/miles/miles/rollout/rm_hub/math_utils.py').read_bytes()).hexdigest(),'boxed_correct_controls_passed':positive,'boxed_wrong_controls_passed':negative,'empty_controls_passed':blank,'irrelevant_controls_passed':len(rows),'model_tokenizer_revision':'7ae557604adf67be50417f59c2c2f167def9a775','inference_runs':0,'optimizer_updates':0,'paid_judge_calls':0,'formatted_example':example}
(root/'preparation.json').write_text(json.dumps(record,indent=2)+'\n');print('validated',json.dumps(record))
