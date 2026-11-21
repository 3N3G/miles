"""Compare candidate concept axes with identical signed per-token edit magnitudes."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForImageTextToText,AutoTokenizer
from pilot import MODEL,REVISION,token_list
from country_swap import features,forward_with_swap


def forward(model,ids,reference,direction):
 counter=[0];details=[]
 def hook(module,args,output):
  visit=counter[0];counter[0]+=1
  if visit!=35:return None
  h=output.float();coefficient=-4*(h@reference).unsqueeze(-1);delta=coefficient*direction
  expected=coefficient.abs().squeeze(-1);error=((delta.norm(dim=-1)-expected).abs()/(expected+1e-6)).max().item();assert error<1e-5,error
  changed=(h+delta).to(output.dtype);actual=(changed.float()-h).norm(dim=-1)
  details.append({'max_precast_relative_norm_error':error,'actual_norms':actual[0].tolist(),'expected_norms':expected[0].tolist(),'last_relative_edit_norm':(delta[0,-1].norm()/h[0,-1].norm()).item()})
  return changed
 handles=[x.register_forward_hook(hook) for x in model.model.language_model.layers]
 try:logits=model(input_ids=ids,use_cache=False).logits[0,-1].float()
 finally:
  for h in handles:h.remove()
 assert counter[0]==48
 return logits,details[0]


def main():
 p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--fits',type=Path,required=True);p.add_argument('--jacobian',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(4)
 tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION);model,info=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'},output_loading_info=True);model.eval().requires_grad_(False);assert not any(info[k] for k in ['missing_keys','mismatched_keys','error_msgs'])
 cases=json.loads(a.cases.read_text());jac=np.load(a.jacobian);vocab={}
 for i in range(model.lm_head.weight.shape[0]):vocab.setdefault(tok.decode([i]).strip().casefold(),[]).append(i)
 native={};valid=[];controls=[];rows=[]
 with torch.inference_mode():
  for case in cases['cases']:
   for prompt in [case['prompt'],case.get('counterfactual_prompt')]:
    if prompt is None or prompt in native:continue
    native[prompt]=forward_with_swap(model,tok(prompt,return_tensors='pt',add_special_tokens=False).input_ids.to('cuda'))[0]
   cf=case.get('counterfactual_prompt');ok=case['kind']!='unrelated' and int(native[case['prompt']].argmax()) in vocab[case['original'].casefold()] and cf is not None and int(native[cf].argmax()) in vocab[case['counterfactual'].casefold()]
   controls.append({'case':case,'valid':ok,'native_top10':token_list(native[case['prompt']],tok),'counterfactual_input_top10':token_list(native[cf],tok) if cf else None})
   if ok:valid.append(case)
  (a.out/'native_controls.json').write_text(json.dumps(controls,indent=2)+'\n')
  for seed in [0,1,2]:
   fs={m:features(model,tok,a.fits,m,seed,jac,cases['source_country'],cases['target_country'],[35]) for m in ['j_unit','raw_unit','ridge_unit']}
   axes={m:f[35]['vector']/f[35]['vector'].norm() for m,f in fs.items()}
   for r in [0,1,2]:
    gen=torch.Generator(device='cuda').manual_seed(1000*r+35);v=torch.randn(2304,generator=gen,device='cuda');axes[f'random_{r}']=v/v.norm()
   for case in valid:
    ids=tok(case['prompt'],return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
    for method,direction in axes.items():
     logits,detail=forward(model,ids,axes['j_unit'],direction)
     if method=='j_unit':
      old=forward_with_swap(model,ids,fs['j_unit'],2)[0]
      # Equivalent formulas can differ at BF16 rounding boundaries. Require identical top one and tiny logits.
      discrepancy=(old-logits).abs().max().item();assert discrepancy<.15,discrepancy;assert int(old.argmax())==int(logits.argmax())
     else:discrepancy=None
     lp=logits.log_softmax(-1);row={'method':method,'fit_seed':seed,'name':case['name'],'prompt':case['prompt'],'counterfactual_top1':int(logits.argmax()) in vocab[case['counterfactual'].casefold()],'target_probability':torch.logsumexp(lp[vocab[case['counterfactual'].casefold()]],0).exp().item(),'top10':token_list(logits,tok),'jacobian_formula_max_logit_discrepancy':discrepancy,'perturbation':detail};rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['perturbation','prompt']}),flush=True)
    (a.out/'interventions.json').write_text(json.dumps(rows,indent=2)+'\n')
 (a.out/'manifest.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'visit':35,'fit_seeds':[0,1,2],'reference_dose':2,'valid_cases':len(valid),'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
if __name__=='__main__':main()
