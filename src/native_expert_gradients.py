"""Development-only routing derivatives. Expert IDs always include their layer."""

import json
import contextlib
import torch
from native_expert import formatted_prompt, render_question


def discover(model, tokenizer, groups, records, fmt, out, offload=False):
    results=[]
    parameter_storage={p.untyped_storage().data_ptr() for p in model.parameters()}
    def pack(tensor):
        if tensor.device.type!='cuda' or tensor.untyped_storage().data_ptr() in parameter_storage:
            return ('original',tensor)
        return ('cpu',tensor.device,tensor.detach().to('cpu',copy=True))
    def unpack(saved):
        return saved[1] if saved[0]=='original' else saved[2].to(saved[1])
    for record in records:
        text,answer=render_question(record)
        ids=tokenizer(formatted_prompt(tokenizer,text,fmt),return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
        captured=[];handles=[];calls={}
        def hook_for(layer):
            def hook(module,args,output):
                logits,weights,indices=output
                calls[layer]=calls.get(layer,0)+1
                if not weights.requires_grad:
                    weights.requires_grad_(True)
                weights.retain_grad()
                captured.append((layer,calls[layer],weights,indices))
                return logits,weights,indices
            return hook
        config=model.model.language_model.config
        for layer in range(config.loop_layer_index_start,config.loop_layer_index_end):
            handles.append(model.model.language_model.layers[layer].mlp.gate.register_forward_hook(hook_for(layer)))
        try:
            context=torch.autograd.graph.saved_tensors_hooks(pack,unpack) if offload else contextlib.nullcontext()
            with context:
                logits=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1].float()
                scores=torch.stack([torch.logsumexp(logits[groups[x]],-1) for x in 'ABCD'])
                objective=scores.log_softmax(-1)['ABCD'.index(answer)]
                objective.backward()
            scores_out=[]
            for layer,visit,weights,indices in captured:
                assert weights.grad is not None
                contribution=weights.detach()*weights.grad.detach()
                signed=torch.zeros(128,device='cuda').scatter_add_(0,indices.reshape(-1),contribution.reshape(-1))
                absolute=torch.zeros(128,device='cuda').scatter_add_(0,indices.reshape(-1),contribution.abs().reshape(-1))
                # Positive weight*gradient predicts that removal reduces the gold objective.
                scores_out.append({'layer_index':layer,'visit':visit,'signed':signed.cpu().tolist(),
                                   'absolute':absolute.cpu().tolist()})
            assert set(calls.values())=={config.loop_num_passes},calls
            results.append({'id':record['id'],'subject':record['subject'],'answer':answer,
                            'objective':'gold four-choice log probability','clean_objective':float(objective.detach()),
                            'scores':scores_out})
            (out/'router_gradients.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps({'event':'gradient','id':record['id'],'peak_gpu_bytes':torch.cuda.max_memory_allocated()}),flush=True)
        finally:
            for handle in handles:handle.remove()
            del captured,logits,objective
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
