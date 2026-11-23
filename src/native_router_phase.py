"""Replay the small native router on CPU and separate pass offsets from content."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from inspect_native_experts import numeric_cue


def router(x,w,b):
    scores=torch.sigmoid(x@w.T)
    indices=(scores+b).topk(8,sorted=False).indices
    weights=scores.gather(-1,indices)
    weights=weights/weights.sum(-1,keepdim=True)*8
    return indices,weights


def overlap(a,b):
    return (a[...,None]==b[...,None,:]).any(-1).float().mean(-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(2)
    records=json.loads((a.run/'inputs.json').read_text())['records']
    original=np.load(a.run/'operator_inputs.npy',mmap_mode='r')
    x=torch.from_numpy(np.array(original[:144])).reshape(144,48,3,2304)
    w=torch.from_numpy(np.fromfile(a.weights/'weight.raw',dtype=np.uint16).copy()).view(torch.bfloat16).float().reshape(128,2304)
    bias=torch.from_numpy(np.fromfile(a.weights/'bias.raw',dtype=np.float32).copy())
    # The pinned loader keeps the correction bias and router computation in FP32.
    route,weights=router(x,w,bias)
    stored_indices=[];stored_weights=[]
    for index in range(144):
        data=np.load(a.run/f'routing_{index:03}.npz');stored_indices.append(data['indices']);stored_weights.append(data['weights'])
    target=torch.from_numpy(np.stack(stored_indices));tw=torch.from_numpy(np.stack(stored_weights))
    same=(route[...,None]==target[...,None,:])
    hit=same.any(-1);replayed=(weights[...,None]*same).sum(-2)
    error=float((replayed-tw).abs().max())
    controls={'top8_set_fraction_exact':float(hit.all(-1).float().mean()),'mean_overlap':float(hit.float().mean()),'max_selected_weight_error':error,
              'source':'checkpoint BF16 weight to FP32, original FP32 correction bias'}
    assert controls['mean_overlap']>.999,controls
    assert error<.01,controls
    means=x[:128].mean((0,1));grand=means.mean(0)
    u,_,_=torch.linalg.svd((means-grand).T,full_matrices=False);basis=u[:,:2]
    validation=x[128:];observed=target[128:];results=[];examples=[]
    for source in range(3):
        for dest in range(3):
            if source==dest:continue
            translated=validation[:,:,source]+means[dest]-means[source]
            predicted,_=router(translated,w,bias)
            constant,_=router(means[dest][None,None].expand(16,48,-1),w,bias)
            naive=overlap(observed[:,:,source],observed[:,:,dest]);pred=overlap(predicted,observed[:,:,dest]);flat=overlap(constant,observed[:,:,dest])
            gain=pred-naive;median=float(gain.median());idx=int((gain.reshape(-1)-median).abs().argmin())
            doc,pos=idx//48,idx%48
            record=records[128+doc]
            examples.append({'source_pass':source+1,'target_pass':dest+1,'id':record['id'],'position':pos+16,
                             'full_input':record['rendered_input'],'token_ids':record['token_ids'],
                             'source_routes':observed[doc,pos,source].tolist(),'target_routes':observed[doc,pos,dest].tolist(),
                             'translated_routes':predicted[doc,pos].tolist(),'overlap_gain':float(gain[doc,pos])})
            results.append({'source_pass':source+1,'target_pass':dest+1,'unchanged_source_overlap':float(naive.mean()),
                            'mean_translated_overlap':float(pred.mean()),'constant_target_mean_overlap':float(flat.mean()),
                            'article_gains':gain.mean(1).tolist()})
    tok=Tokenizer.from_file('/tmp/inkling-operator-tokenizer/tokenizer.json')
    cues=torch.tensor([[numeric_cue(tok.decode([t])) for t in r['token_ids'][16:64]] for r in records[128:144]])
    delta=-((validation-grand)@basis)@basis.T
    normalized=validation+delta
    randoms=[]
    for seed in [0,1,2]:
        generator=torch.Generator().manual_seed(seed)
        q=torch.linalg.qr(torch.randn(2304,2,generator=generator)).Q
        dr=-((validation-grand)@q)@q.T
        dr=dr/dr.norm(dim=-1,keepdim=True).clamp_min(1e-20)*delta.norm(dim=-1,keepdim=True)
        randoms.append((f'random-rank2-seed{seed}',validation+dr))
    edits=[]
    for name,edited in [('native',validation),('remove-fit-pass-mean-span',normalized)]+randoms:
        routes,_=router(edited,w,bias)
        for visit in range(3):
            for expert in [105,45,3]:
                selected=(routes[:,:,visit]==expert).any(-1)
                yes=int((selected&cues).sum());no=int((selected&~cues).sum())
                edits.append({'edit':name,'visit':visit+1,'expert':expert,'with_numeric_cue_selected':yes,'with_numeric_cue':int(cues.sum()),
                              'without_numeric_cue_selected':no,'without_numeric_cue':int((~cues).sum()),
                              'all_expert_overlap_with_native':float(overlap(routes[:,:,visit],observed[:,:,visit]).mean())})
    result={'controls':controls,'transitions':results,'routing_edits':edits,'examples':examples,
            'mean_phase_edit_norm':float(delta.norm(dim=-1).mean()),'mean_native_input_norm':float(validation.norm(dim=-1).mean()),
            'fit_paragraphs':128,'fit_unique_articles':106,'development_articles':16,'development_fit_article_overlap':3,
            'not_full_model_inference':True,'final_states_used':False,
            'router_tensor_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.weights.glob('*.raw')}}
    (a.out/'phase_routing.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(a.out/'fit_phase_statistics.npz',means=means.numpy(),basis=basis.numpy())
    print(json.dumps({k:v for k,v in result.items() if k not in ['examples','routing_edits']},indent=2))


if __name__=='__main__':main()
