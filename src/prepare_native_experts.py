"""Freeze calibration, discovery and reserved question cohorts before inference."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from native_expert import render_question


def main():
    path = Path('evidence/native_expert_inputs.json')
    if path.exists():
        raise FileExistsError(path)
    source = Path('/tmp/inkling-mmlu')
    tok = Tokenizer.from_file('/tmp/inkling-operator-tokenizer/tokenizer.json')
    records, exclusions = [], []
    for subject in ['elementary_mathematics', 'high_school_biology']:
        for split in ['validation', 'test']:
            rows = pq.read_table(source / subject / split / '0000.parquet').to_pylist()
            selected = list(range(24)) if split == 'validation' else np.random.default_rng(1729).permutation(len(rows)).tolist()
            accepted = 0
            for index in selected:
                row = rows[index]
                text, answer = render_question(row)
                length = len(tok.encode(text, add_special_tokens=False).ids)
                identity = f'{subject}-{split}-{index}'
                if length > 480:
                    exclusions.append({'id':identity, 'plain_tokens':length})
                    continue
                if split == 'validation':
                    cohort = 'calibration' if index < 8 else 'discovery'
                else:
                    cohort = 'confirmation' if accepted < 64 else 'replication'
                records.append({'id':identity, 'cohort':cohort, 'subject':subject,
                                'source_split':split, 'source_row':int(index), **row})
                accepted += 1
                if split == 'test' and accepted == 128:
                    break
    previous = json.load(open('/home/ubuntu/inkling-interp-data/shared-operator-v1/inputs.json'))['records']
    text_records = [r for r in previous if r['split']=='validation']
    freq = np.zeros(128)
    for i, r in enumerate(previous):
        if r['split'] == 'fit':
            indices=np.load(f'/home/ubuntu/inkling-interp-data/shared-operator-v1/routing_{i:03}.npz')['indices']
            freq += np.bincount(indices.reshape(-1), minlength=128)
    matched = sorted((e for e in range(128) if e not in [105,45,3]),key=lambda e:(abs(freq[e]-freq[105]),e))[:3]
    result={'model_revision':'ed40e7da625f594fc9c4990f5f66be24873c22d7',
            'dataset':'cais/mmlu','dataset_revision':'c30699e8356da336a370243923dbaf21066bb9fe',
            'parquet_revision':(source/'pin.txt').read_text(),
            'file_sha256':{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.parquet')},
            'selection_seed':1729,'answer_order_seeds':[0,1,2], 'layer_index':11,
            'frequency_matched_experts':matched,'old_fit_route_counts':freq.astype(int).tolist(),
            'records':records,'excluded_before_inference':exclusions,'development_text':text_records}
    path.write_text(json.dumps(result,indent=2)+'\n')
    from collections import Counter
    print(dict(Counter((r['subject'],r['cohort']) for r in records)))
    print('Matched experts',matched,'counts',freq[[105]+matched].tolist())
    print('Example',json.dumps(records[0]))


if __name__=='__main__':main()
