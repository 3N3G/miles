"""Independent-text confirmation of the apparent three-pass orbit."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForImageTextToText,AutoTokenizer
from pilot import MODEL,REVISION,token_list
from long_loop import forward


def main():
 p=argparse.ArgumentParser();p.add_argument('--calibration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(4)
 tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
 model,info=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'},output_loading_info=True)
 model.eval().requires_grad_(False);assert not any(info[k] for k in ['missing_keys','mismatched_keys','error_msgs']);assert model.model.language_model.config.loop_num_passes==3
 data=json.loads(a.calibration.read_text());rows=[]
 with torch.inference_mode():
  for i,text in enumerate(data['texts'][128:136]):
   ids=tok(text,return_tensors='pt',add_special_tokens=False).input_ids[:,:64].to('cuda');assert ids.shape[-1]==64
   native,short,e3=forward(model,ids,3);extended,states,e64=forward(model,ids,64);assert np.array_equal(short,states[:3])
   np.savez_compressed(a.out/f'states_{i:02}.npz',states=states.astype(np.float16),token_ids=ids.cpu().numpy())
   norm=states/np.sqrt(np.mean(states**2,axis=-1,keepdims=True)+1e-6)
   bylag={};raw={};last={}
   for lag in range(1,9):
    bylag[str(lag)]=np.sqrt(np.mean((norm[48:]-norm[48-lag:64-lag])**2,axis=(1,2,3))).tolist()
    raw[str(lag)]=np.sqrt(np.mean((states[48:]-states[48-lag:64-lag])**2,axis=(1,2,3))).tolist()
    last[str(lag)]=np.sqrt(np.mean((norm[48:,:,-1]-norm[48-lag:64-lag,:,-1])**2,axis=(1,2))).tolist()
   row={'calibration_index':128+i,'rendered_input':tok.decode(ids[0]),'input_ids':ids[0].tolist(),'native_top10':token_list(native,tok),'extended_top10':token_list(extended,tok),'native_head_errors':[e3,e64],'first_three_states_exact':True,'tail_normalized_distances_by_lag':bylag,'tail_raw_rms_distances_by_lag':raw,'tail_last_token_normalized_distances_by_lag':last}
   rows.append(row);(a.out/'confirmation.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({'event':'paragraph','index':128+i,'lag1':np.mean(bylag['1']),'lag3':np.mean(bylag['3']),'ratio':np.mean(bylag['3'])/np.mean(bylag['1']),'elapsed':time.monotonic()-start}),flush=True)
 (a.out/'manifest.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'paragraph_indices':list(range(128,136)),'native_passes':3,'extended_passes':64,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
if __name__=='__main__':main()
