"""Freeze one new paragraph per WikiText test article, excluding previous articles."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def main():
    dest=Path('evidence/native_expert_text.json');assert not dest.exists()
    source=Path('/tmp/inkling-operator-source/wikitext-2-raw-v1/test/0000.parquet')
    prior=Path('/home/ubuntu/inkling-interp-data/shared-operator-v1')
    articles=set(json.load(open(prior/'article_provenance.json'))['article_by_calibration_id'].values())
    articles.update(r.get('article') for r in json.load(open(prior/'inputs.json'))['records'])
    texts=pq.read_table(source)['text'].to_pylist();tok=Tokenizer.from_file('/tmp/inkling-operator-tokenizer/tokenizer.json')
    candidates=[];article=None
    for index,text in enumerate(texts):
        heading=text.strip()
        if heading.startswith('= ') and not heading.startswith('= ='):article=heading
        if article in articles or heading.startswith('=') or len(text)<500:continue
        ids=tok.encode(text,add_special_tokens=False).ids
        if len(ids)<128:continue
        candidates.append((index,text,ids[:128],article))
    rng=np.random.default_rng(1729);rng.shuffle(candidates);selected=[];seen=set()
    for index,text,ids,article in candidates:
        if article in seen:continue
        seen.add(article);selected.append({'id':f'text-confirmation-{len(selected):03}','source_row':index,'article':article,
                                         'cohort':'text_confirmation','text':text,'token_ids':ids,'rendered_input':tok.decode(ids)})
        if len(selected)==48:break
    assert len(selected)==48,len(selected)
    dest.write_text(json.dumps({'dataset':'Salesforce/wikitext','parquet_revision':'3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e',
                              'source_split':'test','subset':'wikitext-2-raw-v1','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                              'selection_seed':1729,'distinct_articles':48,'prior_article_overlap':0,'records':selected},indent=2)+'\n')
    print('Frozen48 distinct articles. Example:',json.dumps(selected[0]))


if __name__=='__main__':main()
