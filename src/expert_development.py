"""Repair the derivative screen, validate its candidates and observe native units."""

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from native_expert import aliases,formatted_prompt,letter_metrics,render_question
from native_expert_confirmation import run,make_masks,write
from native_expert_gradients import discover


def choose_candidates(rows):
    data=defaultdict(lambda:defaultdict(list))
    for row in rows:
        summed=defaultdict(lambda:np.zeros(128))
        for score in row['scores']:summed[score['layer_index']]+=np.array(score['signed'])
        for layer,values in summed.items():
            for expert,value in enumerate(values):data[(layer,expert)][row['subject']].append(float(value))
    ranked=[]
    for (layer,expert),values in data.items():
        assert all(len(values[s])==16 for s in ['elementary_mathematics','high_school_biology'])
        math=np.mean(values['elementary_mathematics']);bio=np.mean(values['high_school_biology'])
        ranked.append({'layer_index':layer,'expert':expert,'math_gradient':float(math),'biology_gradient':float(bio),'contrast':float(math-bio)})
    ranked.sort(key=lambda r:(-r['contrast'],r['layer_index'],r['expert']))
    return [r for r in ranked if r['math_gradient']>0][:2],ranked


def make_spec(candidates):
    conditions=[]
    for c in candidates:
        target={'layer_index':c['layer_index'],'expert':c['expert']};key=f"L{c['layer_index']+1}-E{c['expert']}"
        conditions.append({'name':key+'-all','targets':[target]})
        for visit in [1,2,3]:conditions.append({'name':key+f'-pass{visit}','targets':[{**target,'visits':[visit]}]})
        conditions.append({'name':key+'-renormalized','renormalize':True,'targets':[target]})
        conditions.append({'name':key+'-random','kind':'matched_random','targets':[target]})
    return {'format':'chat_text','conditions':conditions,'candidates':candidates}


def main():
    from transformers import AutoModelForImageTextToText,AutoTokenizer
    from pilot import MODEL,REVISION,token_list
    from capture_native_neurons import capture
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--operator-inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);started=time.time()
    inputs=json.loads(a.inputs.read_text());records=[r for r in inputs['records'] if r['cohort']=='discovery'];assert len(records)==32
    write(a.out/'frozen_inputs.json',inputs)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION);groups=aliases(tok)
    model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map={'':'cuda:0'}).eval();model.requires_grad_(False)
    text,answer=render_question(records[0]);ids=tok(formatted_prompt(tok,text,'chat_text'),return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
    baseline,_=run(model,ids,{}, {},1729)
    discover(model,tok,groups,records,'chat_text',a.out,offload=True)
    replay,_=run(model,ids,{}, {},1729);assert torch.equal(baseline,replay),'Derivatives modified native logits'
    assert all(p.grad is None for p in model.parameters())
    write(a.out/'gradient_controls.json',{'native_logits_unchanged_after_gradients':True,'pretrained_weight_gradients_absent':True,
        'core_zero_based_indices':[model.model.language_model.config.loop_layer_index_start,model.model.language_model.config.loop_layer_index_end],
        'passes':model.model.language_model.config.loop_num_passes})
    grad=json.loads((a.out/'router_gradients.json').read_text());candidates,ranking=choose_candidates(grad)
    write(a.out/'candidate_ranking.json',ranking);spec=make_spec(candidates);write(a.out/'frozen_spec.json',spec)
    print(json.dumps({'event':'candidates_frozen','candidates':candidates,'elapsed':time.time()-started}),flush=True)
    rows=[]
    for index,record in enumerate(records):
        text,answer=render_question(record);rendered=formatted_prompt(tok,text,spec['format'])
        ids=tok(rendered,return_tensors='pt',add_special_tokens=False).input_ids.to('cuda');masks=make_masks(tok,rendered,record['question'],ids)
        baseline,_=run(model,ids,{},masks,1729)
        if index==0:
            first=spec['conditions'][0];deleted,_=run(model,ids,first,masks,1729)
            output_deleted,_=run(model,ids,{**first,'kind':'output_delete'},masks,1729)
            assert torch.equal(deleted,output_deleted),'Two native deletion implementations differ'
            absent,_=run(model,ids,{'targets':[{'layer_index':11,'expert':128}]},masks,1729)
            assert torch.equal(baseline,absent)
            write(a.out/'native_edit_controls.json',{'router_vs_output_deletion_error':0,'absent_expert_error':0})
        for cond in [{'name':'clean'}]+spec['conditions']:
            logits,audit=(baseline,[]) if cond['name']=='clean' else run(model,ids,cond,masks,1729+index)
            rows.append({'id':record['id'],'subject':record['subject'],'condition':cond['name'],'answer':answer,
                         'rendered_input':rendered,'token_ids':ids[0].tolist(),'audit':audit,
                         **letter_metrics(logits,groups,answer),'top_tokens':token_list(logits,tok,k=5)})
        write(a.out/'results.json',rows)
        print(json.dumps({'event':'candidate_validation','id':record['id'],'elapsed':time.time()-started}),flush=True)
    operator_inputs=json.loads(a.operator_inputs.read_text())
    capture(model,[r for r in operator_inputs['records'] if r['split'] in ['fit','validation']],a.out/'native_neurons',experts=(105,45,3,89,41,90))
    write(a.out/'manifest.json',{'status':'completed','wall_seconds':time.time()-started,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                               'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'model':MODEL,'revision':REVISION,
                               'input_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest()})


if __name__=='__main__':main()
