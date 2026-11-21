"""Single-pass causal transfer of the third-pass country Jacobian direction."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForImageTextToText,AutoTokenizer
from pilot import MODEL,REVISION,token_list
from country_swap import features,forward_with_swap


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--fits',type=Path,required=True);p.add_argument('--jacobian',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model,info=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'},output_loading_info=True)
    model.eval().requires_grad_(False);assert not any(info[k] for k in ['missing_keys','mismatched_keys','error_msgs'])
    cf=json.loads(a.cases.read_text());jac=np.load(a.jacobian);token_map={}
    for i in range(model.lm_head.weight.shape[0]):token_map.setdefault(tok.decode([i]).strip().casefold(),[]).append(i)
    native={};controls=[];valid=[];rows=[]
    with torch.inference_mode():
      for case in cf['cases']:
        for prompt in [case['prompt'],case.get('counterfactual_prompt')]:
          if prompt is None or prompt in native:continue
          ids=tok(prompt,return_tensors='pt',add_special_tokens=False).input_ids.to('cuda');native[prompt]=forward_with_swap(model,ids)[0]
        baseline=int(native[case['prompt']].argmax()) in token_map[case['original'].casefold()]
        counter=int(native[case['counterfactual_prompt']].argmax()) in token_map[case['counterfactual'].casefold()] if case.get('counterfactual_prompt') else False
        controls.append({'case':case,'baseline_valid':baseline,'counterfactual_input_valid':counter,'original_top10':token_list(native[case['prompt']],tok),'counterfactual_input_top10':token_list(native[case['counterfactual_prompt']],tok) if case.get('counterfactual_prompt') else None})
        if case['kind']!='unrelated' and baseline and counter:valid.append(case)
      (a.out/'native_controls.json').write_text(json.dumps(controls,indent=2)+'\n')
      for site in [11,23,35]:
       for seed in [0,1,2]:
        vectors=features(model,tok,a.fits,'j_unit',seed,jac,cf['source_country'],cf['target_country'],[site])
        for case in valid:
          ids=tok(case['prompt'],return_tensors='pt',add_special_tokens=False).input_ids.to('cuda');base=native[case['prompt']];target=token_map[case['counterfactual'].casefold()]
          logits,_=forward_with_swap(model,ids,vectors,0);assert torch.equal(logits,base)
          for dose in [1,2]:
            for random_seed in [None,0,1,2]:
              logits,changes=forward_with_swap(model,ids,vectors,dose,random_seed);lp=logits.log_softmax(-1)
              row={'method':'j_unit','site':site,'fit_seed':seed,'name':case['name'],'kind':case['kind'],'prompt':case['prompt'],'dose':dose,'random_seed':random_seed,'baseline_valid':True,'counterfactual_input_valid':True,'counterfactual_top1':int(logits.argmax()) in target,'unchanged_top1':int(logits.argmax())==int(base.argmax()),'target_probability':torch.logsumexp(lp[target],0).exp().item(),'source_probability':torch.logsumexp(lp[token_map[case['original'].casefold()]],0).exp().item(),'top5':token_list(logits,tok,k=5),'changes':changes}
              rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['changes','prompt']}),flush=True)
          (a.out/'interventions.json').write_text(json.dumps(rows,indent=2)+'\n')
    (a.out/'manifest.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'sites':[11,23,35],'row_source_visit':35,'fit_seeds':[0,1,2],'original_cases':len(cf['cases']),'valid_cases':len(valid),'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
if __name__=='__main__':main()
