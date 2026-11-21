"""Execute the single frozen mean/scale versus raw held-out comparison."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from pilot import MODEL, REVISION, token_list
from capture_transport import capture
from fit_transport import VISITS, prepare_items, score_predictions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--items',type=Path,required=True)
    p.add_argument('--fits',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    t=time.monotonic(); torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model,info=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'},output_loading_info=True)
    model.eval().requires_grad_(False)
    assert not any(info[k] for k in ['missing_keys','mismatched_keys','error_msgs'])
    items=json.loads(a.items.read_text())['items']
    assert len(items)==43
    data={k:[] for k in ['active','mixed','target','logits']}; outputs=[]
    with torch.inference_mode():
        for item in items:
            ids=tok(item['prompt'],return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
            assert ids.shape[-1]<=256
            states,target,logits,error=capture(model,ids,[-1],verify=True)
            for k in ['active','mixed']: data[k].append(states[k][0])
            data['target'].append(target[0]); data['logits'].append(logits[0])
            generated=[]; next_logits=torch.tensor(logits[0],device='cuda')
            for step in range(12):
                if step: next_logits=model(input_ids=ids,use_cache=False).logits[0,-1].float()
                token=int(next_logits.argmax()); generated.append(token)
                if token==200006: break
                ids=torch.cat([ids,ids.new_tensor([[token]])],dim=-1)
            row={'item':item,'native_logit_error':error,'final_tokens':token_list(torch.tensor(logits[0]),tok),'generated_ids':generated,'generated_text':tok.decode(generated)}
            outputs.append(row)
            (a.out/'development_outputs.json').write_text(json.dumps(outputs,indent=2)+'\n')
            print(json.dumps({'event':'heldout_native','name':item['name'],'prompt':item['prompt'],'generated_text':row['generated_text'],'native_logit_error':error}),flush=True)
        arrays={k:np.array(v) for k,v in data.items()}
        np.savez_compressed(a.out/'development_states.npz',**arrays)
        tokens=[tok.decode([i]) for i in range(model.lm_head.weight.shape[0])]
        mask=torch.tensor([any(c.isalpha() or c.isdigit() for c in s) for s in tokens],device='cuda')
        prepared=prepare_items(outputs,tokens)
        native=torch.tensor(arrays['logits'],device='cuda')
        states=torch.tensor(arrays['active'],device='cuda').float()
        rows=[]; offsets={}
        for seed in [0,1,2]:
            maps=np.load(a.fits/f'maps_active_seed_{seed}.npz')
            for visit in VISITS:
                prefix=f'active_v{visit}_ridge_'
                stats={k:torch.tensor(maps[prefix+k],device='cuda') for k in ['mx','sx','my','sy']}
                h=states[:,visit]
                ratio=stats['sy']/stats['sx']
                corrected=stats['my']+ratio*(h-stats['mx'])
                offset=stats['my']/ratio-stats['mx']
                assert torch.allclose(corrected/ratio,h+offset,atol=2e-5,rtol=2e-5)
                offsets[f'seed_{seed}_v{visit}']=offset.cpu().numpy()
                for method,pred in [('raw',h),('mean_scale',corrected)]:
                    rows.extend(score_predictions(pred,native,prepared,tokens,mask,model.lm_head.weight,model.model.language_model.norm.weight.float(),{'mode':'active','method':method,'seed':seed,'visit':visit}))
        (a.out/'readouts.json').write_text(json.dumps(rows,indent=2)+'\n')
        (a.out/'diagnostics.json').write_text('[]\n')
        np.savez_compressed(a.out/'readout_offsets.npz',**offsets)
        raw={}
        for row in rows:
            if row['method']!='raw': continue
            key=(row['name'],row['visit'],row['filtered'])
            ids=[x['id'] for x in row['top10']]
            if key in raw: assert raw[key]==ids
            raw[key]=ids
    (a.out/'manifest.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'items':len(items),'visits':VISITS,'seeds':[0,1,2],'primary':'seed0, active, filtered mean_scale versus raw, capable_unspoken and single-token eligible','elapsed_seconds':time.monotonic()-t},indent=2)+'\n')
    print(json.dumps({'event':'complete','elapsed_seconds':time.monotonic()-t}),flush=True)

if __name__=='__main__': main()
