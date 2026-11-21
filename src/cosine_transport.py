"""Separate the final-state prior from centered vocabulary-direction readout."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer
from pilot import MODEL,REVISION
from fit_transport import VISITS,prepare_items,score_predictions


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--fits',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);t=time.monotonic();torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    with safe_open(hf_hub_download(MODEL,'model-00001-of-00001.safetensors',revision=REVISION),framework='pt',device='cpu') as f:
        folded=f.get_tensor('model.llm.unembed.weight')[:200058].float().to('cuda')*f.get_tensor('model.llm.norm.weight').float().to('cuda')
    tokens=[tok.decode([i]) for i in range(len(folded))];mask=torch.tensor([any(c.isalpha() or c.isdigit() for c in s) for s in tokens],device='cuda')
    items=prepare_items(json.loads((a.capture/'development_outputs.json').read_text()),tokens)
    dev=np.load(a.capture/'development_states.npz');native=torch.tensor(dev['logits'],device='cuda');hd=torch.tensor(dev['active'],device='cuda').float()
    rows=[];diagnostics=[];ones=torch.ones(2304,device='cuda')
    with torch.inference_mode():
      for seed in [0,1,2]:
        maps=np.load(a.fits/f'maps_active_seed_{seed}.npz')
        for visit in VISITS:
          prefix=f'active_v{visit}_ridge_';mx=torch.tensor(maps[prefix+'mx'],device='cuda');matrix=torch.tensor(maps[prefix+'matrix'],device='cuda');x=hd[:,visit]-mx
          for basis in ['raw','ridge']:
            vectors=folded if basis=='raw' else folded@matrix.T
            norms=vectors.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            for cosine in [False,True]:
              head=(vectors/norms if cosine else vectors).bfloat16()
              if cosine:assert ((vectors/norms).norm(dim=-1)-1).abs().max()<1e-5
              method='centered_'+basis+('_cosine' if cosine else '')
              meta={'mode':'active','method':method,'seed':seed,'visit':visit}
              rows.extend(score_predictions(x,native,items,tokens,mask,head,ones,meta))
              diagnostics.append({**meta,'row_norm_min':norms.min().item(),'row_norm_median':norms.median().item(),'row_norm_max':norms.max().item()})
            if basis=='ridge':del vectors
          (a.out/'readouts.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({'seed':seed,'visit':visit,'elapsed':time.monotonic()-t}),flush=True)
    (a.out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n');(a.out/'manifest.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'score_type':'Centered vocabulary inner products/cosines. Fixed RMS and softcap preserve rankings except BF16 rounding. KL is not calibrated and must not be used to compare faithfulness.','elapsed_seconds':time.monotonic()-t},indent=2)+'\n')
if __name__=='__main__':main()
