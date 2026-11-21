"""Observe whether a country-probe exchange persists through later loop passes."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForImageTextToText,AutoTokenizer
from pilot import MODEL,REVISION,token_list
from country_swap import features,forward_with_swap


def observed_forward(model,ids,probe,edits=None,random_seed=None):
    counter=[0];trace=[];snapshots={}
    def hook(module,args,output):
        visit=counter[0];counter[0]+=1
        if visit not in [11,23,35]:return None
        h=output.float();before=(h[0,-1]@probe/(h[0,-1].norm()*probe.norm())).item();changed=None
        if edits is not None and visit in edits:
            v=edits[visit]['vector'];b=edits[visit]['bias'];delta=-4*(h@v+b).unsqueeze(-1)*v/v.square().sum()
            if random_seed is not None:
                gen=torch.Generator(device='cuda').manual_seed(1000*random_seed+visit);direction=torch.randn(v.shape,generator=gen,device='cuda');direction/=direction.norm();delta=delta.norm(dim=-1,keepdim=True)*direction
            changed=(h+delta).to(output.dtype);h=changed.float()
        after=(h[0,-1]@probe/(h[0,-1].norm()*probe.norm())).item()
        trace.append({'visit':visit,'probe_cosine_before':before,'probe_cosine_after':after,'edited':changed is not None})
        snapshots[visit]=h[0].cpu().numpy()
        return changed
    handles=[layer.register_forward_hook(hook) for layer in model.model.language_model.layers]
    try:logits=model(input_ids=ids,use_cache=False).logits[0,-1].float()
    finally:
        for h in handles:h.remove()
    assert counter[0]==48
    return logits,trace,snapshots


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--fits',type=Path,required=True);p.add_argument('--third',type=Path,required=True);p.add_argument('--early',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION);model,info=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'},output_loading_info=True)
    model.eval().requires_grad_(False);assert not any(info[k] for k in ['missing_keys','mismatched_keys','error_msgs'])
    case_file=json.loads(a.cases.read_text());third=np.load(a.third);early=np.load(a.early);src,tgt=case_file['source_country'],case_file['target_country'];vocab={}
    for i in range(model.lm_head.weight.shape[0]):vocab.setdefault(tok.decode([i]).strip().casefold(),[]).append(i)
    third_features=features(model,tok,a.fits,'j_unit',0,third,src,tgt,[11,23,35]);early_features=features(model,tok,a.fits,'j_native',0,early,src,tgt,[11,23]);probe=third_features[35]['vector'];rows=[];controls=[];all_snapshots={}
    with torch.inference_mode():
      for ci,case in enumerate(case_file['cases']):
        ids=tok(case['prompt'],return_tensors='pt',add_special_tokens=False).input_ids.to('cuda');native=forward_with_swap(model,ids)[0]
        cf_prompt=case.get('counterfactual_prompt');counter=forward_with_swap(model,tok(cf_prompt,return_tensors='pt',add_special_tokens=False).input_ids.to('cuda'))[0] if cf_prompt else None
        valid=case['kind']!='unrelated' and int(native.argmax()) in vocab[case['original'].casefold()] and counter is not None and int(counter.argmax()) in vocab[case['counterfactual'].casefold()]
        controls.append({'case':case,'valid':valid,'native_top10':token_list(native,tok),'counterfactual_input_top10':token_list(counter,tok) if counter is not None else None})
        if not valid:continue
        clean,trace,base_states=observed_forward(model,ids,probe);assert torch.equal(clean,native)
        conditions=[('clean',None,None),('third_v11',{11:third_features[11]},None),('third_v23',{23:third_features[23]},None),('third_v35',{35:third_features[35]},None),('third_all',third_features,None),('own_v11',{11:early_features[11]},None),('own_v23',{23:early_features[23]},None),('random_v11',{11:third_features[11]},0)]
        for name,edits,random_seed in conditions:
          logits,trace,snapshots=observed_forward(model,ids,probe,edits,random_seed)
          if name in ['third_v11','third_v23','third_v35']:
            reference=forward_with_swap(model,ids,edits,2)[0];assert torch.equal(reference,logits)
          for step in trace:
            site=step['visit'];x=snapshots[site];b=base_states[site];xn=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6);bn=b/np.sqrt(np.mean(b*b,axis=-1,keepdims=True)+1e-6);step['normalized_bank_rms_distance_to_clean']=float(np.sqrt(np.mean((xn-bn)**2)))
            all_snapshots[f'case{ci}_{name}_v{site}']=x.astype(np.float16)
          lp=logits.log_softmax(-1);row={'name':case['name'],'prompt':case['prompt'],'condition':name,'trace':trace,'counterfactual_top1':int(logits.argmax()) in vocab[case['counterfactual'].casefold()],'target_probability':torch.logsumexp(lp[vocab[case['counterfactual'].casefold()]],0).exp().item(),'top10':token_list(logits,tok)};rows.append(row);print(json.dumps(row),flush=True)
        (a.out/'traces.json').write_text(json.dumps(rows,indent=2)+'\n')
    np.savez_compressed(a.out/'activation_snapshots.npz',**all_snapshots);(a.out/'native_controls.json').write_text(json.dumps(controls,indent=2)+'\n');(a.out/'manifest.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'fit_seed':0,'dose':2,'probe_source_visit':35,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
if __name__=='__main__':main()
