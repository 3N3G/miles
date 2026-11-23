"""Freeze unseen text articles for a native unit-to-block-routing intervention."""

import hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def main():
    dest=Path('evidence/unit_routing_inputs.json');assert not dest.exists()
    old=Path('/home/ubuntu/inkling-interp-data/shared-operator-v1')
    excluded=set(json.load(open(old/'article_provenance.json'))['article_by_calibration_id'].values())
    excluded.update(r.get('article') for r in json.load(open(old/'inputs.json'))['records'])
    excluded.update(r.get('article') for r in json.load(open('evidence/native_expert_text.json'))['records'])
    source=Path('/tmp/inkling-operator-source/wikitext-2-raw-v1/validation/0000.parquet')
    texts=pq.read_table(source)['text'].to_pylist();tok=Tokenizer.from_file('/tmp/inkling-operator-tokenizer/tokenizer.json')
    candidates=[];article=None
    for index,text in enumerate(texts):
        heading=text.strip()
        if heading.startswith('= ') and not heading.startswith('= ='):article=heading
        if not article or article in excluded or heading.startswith('=') or len(text)<500:continue
        ids=tok.encode(text,add_special_tokens=False).ids
        if len(ids)>=128:candidates.append((index,article,ids[:128]))
    np.random.default_rng(2718).shuffle(candidates);seen=set();records=[]
    for index,article,ids in candidates:
        if article in seen:continue
        seen.add(article);records.append({'id':f'unit-routing-{len(records):03}','article':article,'source_row':index,'token_ids':ids,'rendered_input':tok.decode(ids)})
        if len(records)==16:break
    assert len(records)==16
    dest.write_text(json.dumps({'dataset':'Salesforce/wikitext','subset':'wikitext-2-raw-v1','split':'validation',
        'parquet_revision':'3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e','parquet_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'selection_seed':2718,'prior_article_overlap':0,'records':records},indent=2)+'\n')
    print('Sixteen unseen articles. Example:',json.dumps(records[0]))


if __name__=='__main__':main()
