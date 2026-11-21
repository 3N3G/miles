"""Measure the actual full recurrent trajectory beyond the trained pass count."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForImageTextToText,AutoTokenizer
from pilot import MODEL,REVISION,token_list


def forward(model,ids,count,retain=False):
    llm=model.model.language_model
    old=llm.config.loop_num_passes
    seen=[0]; first=[]; second=[]; final=[]
    def hook(module,args,output):
        v=seen[0]; seen[0]+=1
        if v>=6 and v<6+12*count:
            phase=(v-6)%12
            if phase==5: first.append(output.detach().float().cpu().numpy())
            if phase==11: second.append(output.detach().float().cpu().numpy())
    def target_hook(module,args,output): final.append(output[0])
    handles=[x.register_forward_hook(hook) for x in llm.layers]
    handles.append(llm.final_input.register_forward_hook(target_hook))
    try:
        llm.config.loop_num_passes=count
        logits=model(input_ids=ids,use_cache=False).logits[0,-1].float()
        reconstructed=model.logit_softcap(model.lm_head(llm.norm(final[0]).to(model.lm_head.weight.dtype)).float())[0,-1]
        error=(reconstructed-logits).abs().max().item(); assert error==0,error
    finally:
        llm.config.loop_num_passes=old
        for h in handles:h.remove()
    assert seen[0]==12+12*count and len(first)==len(second)==count
    states=np.stack([np.array(first)[:,0],np.array(second)[:,0]],axis=1)
    return logits,states,error


def main():
    p=argparse.ArgumentParser();p.add_argument('--items',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);t=time.monotonic();torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model,info=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'},output_loading_info=True)
    model.eval().requires_grad_(False);llm=model.model.language_model
    assert not any(info[k] for k in ['missing_keys','mismatched_keys','error_msgs'])
    assert llm.config.loop_num_passes==3
    items=json.loads(a.items.read_text())['items'];rows=[];counts=[1,2,3,4,8,16,32]
    with torch.inference_mode():
        for i,item in enumerate(items):
            ids=tok(item['prompt'],return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
            by_count={};native_states=None
            for count in counts:
                logits,states,error=forward(model,ids,count)
                if count==3: native_states=states.copy()
                if count==32:
                    assert np.array_equal(states[:3],native_states)
                    np.savez_compressed(a.out/f'states_{i:02}.npz',states=states.astype(np.float16),token_ids=ids.cpu().numpy())
                    normalized=states/np.sqrt(np.mean(states**2,axis=-1,keepdims=True)+1e-6)
                    one=np.sqrt(np.mean((normalized[1:]-normalized[:-1])**2,axis=(1,2,3)))
                    two=np.sqrt(np.mean((normalized[2:]-normalized[:-2])**2,axis=(1,2,3)))
                    raw=np.sqrt(np.mean((states[1:]-states[:-1])**2,axis=(1,2,3)))/np.sqrt(np.mean(states[:-1]**2,axis=(1,2,3)))
                by_count[str(count)]={'top10':token_list(logits,tok),'argmax':int(logits.argmax()),'native_head_error':error}
                print(json.dumps({'event':'stopping_count','name':item['name'],'prompt':item['prompt'],'passes':count,'top5':by_count[str(count)]['top10'][:5],'elapsed':time.monotonic()-t}),flush=True)
            row={'item':item,'stops':by_count,'one_step_normalized_rms_distance':one.tolist(),'two_step_normalized_rms_distance':two.tolist(),'one_step_raw_relative_rms_distance':raw.tolist(),'first_three_states_exact':True}
            rows.append(row);(a.out/'trajectories.json').write_text(json.dumps(rows,indent=2)+'\n')
    (a.out/'manifest.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'items':len(items),'stopping_counts':counts,'original_loop_passes':3,'elapsed_seconds':time.monotonic()-t},indent=2)+'\n')

if __name__=='__main__':main()
