"""Reserve new articles before reading the native unit atlas's candidate identities."""

import hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def main():
    dest=Path('evidence/unit_atlas_test_inputs.json');assert not dest.exists()
    old=Path('/home/ubuntu/inkling-interp-data/shared-operator-v1')
    excluded=set(json.loads((old/'article_provenance.json').read_text())['article_by_calibration_id'].values())
    for path in [old/'inputs.json',Path('evidence/native_expert_text.json'),Path('evidence/unit_routing_inputs.json')]:
        excluded.update(r.get('article') for r in json.loads(path.read_text())['records'])
    prior_prefixes=set()
    for path in [old/'inputs.json',Path('evidence/native_expert_text.json'),Path('evidence/unit_routing_inputs.json')]:
        prior_prefixes.update(tuple(r['token_ids'][:64]) for r in json.loads(path.read_text())['records'] if 'token_ids' in r)
    source=Path('/tmp/inkling-operator-source/wikitext-2-raw-v1/train/0000.parquet')
    texts=pq.read_table(source)['text'].to_pylist();tok=Tokenizer.from_file('/tmp/inkling-operator-tokenizer/tokenizer.json')
    candidates=[];article=None
    for index,text in enumerate(texts):
        heading=text.strip()
        if heading.startswith('= ') and not heading.startswith('= ='):article=heading
        if not article or article in excluded or heading.startswith('=') or len(text)<500:continue
        ids=tok.encode(text,add_special_tokens=False).ids
        if len(ids)>=128 and tuple(ids[:64]) not in prior_prefixes:candidates.append((index,article,ids[:128]))
    np.random.default_rng(31415).shuffle(candidates);seen=set();records=[]
    for index,article,ids in candidates:
        if article in seen:continue
        seen.add(article);records.append({'id':f'unit-atlas-test-{len(records):03}','article':article,'source_row':index,'token_ids':ids,'rendered_input':tok.decode(ids)})
        if len(records)==32:break
    assert len(records)==32,len(records)
    dest.write_text(json.dumps({'dataset':'Salesforce/wikitext','subset':'wikitext-2-raw-v1','split':'train',
        'parquet_revision':'3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e','parquet_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'selection_seed':31415,'prior_article_overlap':0,'prior_64_token_prefix_overlap':0,'records':records},indent=2)+'\n')
    print(json.dumps({'article_count':32,'example':records[0]},indent=2))


if __name__=='__main__':main()
