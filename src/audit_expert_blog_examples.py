"""Check blog figures and displayed values against the completed run artifacts."""

import argparse,hashlib,json,re
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--examples',type=Path,default=Path('evidence/expert_blog_examples.json'))
    p.add_argument('--article',type=Path,default=Path('reports/EXPERT_CASE_STUDIES.md'));a=p.parse_args()
    r=json.loads(a.examples.read_text());article=a.article.read_text();checked=[]
    for name,digest in r['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    def number(name,value,places=3,signed=False):
        word=format(value,('+' if signed else '')+'.'+str(places)+'f').replace('-','−')
        assert word in article,(name,word)
        checked.append({'label':name,'source_value':value,'display':word})
    first=r['original_normalization_case']
    for visit in ['1','3']:
        z=first['reader_effects'][visit]
        number('native_reader_pass'+visit,100*z['native_block2_weight'])
        number('deleted_reader_pass'+visit,100*z['deleted_block2_weight'])
        number('reader_change_pass'+visit,100*(z['deleted_block2_weight']-z['native_block2_weight']),4,True)
    def factors(example,name):
        for visit in ['pass1','pass3']:
            z=example[visit];g=z['raw_gate'];u=z['raw_up'];h=z['raw_activation']
            reconstructed=4*np.tanh(g/4)/(1+np.exp(-g))*25*np.tanh(u/25)
            assert abs(h-reconstructed)<=max(1e-6,abs(h)*.005),(name,visit,h,reconstructed)
            assert np.sign(h)==np.sign(g*u),(name,visit)
    factors(first,'original')
    gate=r['gate_case']['example'];factors(gate,'gate')
    for visit in ['pass1','pass3']:
        for key in ['raw_gate','raw_up','raw_activation']:number('gate_example_'+visit+'_'+key,gate[visit][key],3,True)
    assert gate['pass1']['raw_gate']*gate['pass3']['raw_gate']<0
    assert gate['pass1']['raw_up']*gate['pass3']['raw_up']>0
    assert first['pass1']['raw_gate']*first['pass3']['raw_gate']>0
    assert first['pass1']['raw_up']*first['pass3']['raw_up']<0
    for case in [first,gate]:
        for visit in ['pass1','pass3']:
            g=case[visit]['raw_gate'];u=case[visit]['raw_up']
            activation=lambda v:4*np.tanh(g/4)/(1+np.exp(-g))*25*np.tanh(v/25)
            assert np.isclose(activation(-u),-activation(u),rtol=1e-12)
    opposite=r['opposite_activity_case']['example'];factors(opposite,'opposite')
    for visit in ['pass1','pass3']:number('opposite_'+visit,opposite[visit]['raw_activation'],3,True)
    for i,z in enumerate(r['mixed_activity_case']['examples']):
        factors(z,'mixed'+str(i))
        for visit in ['pass1','pass3']:number('mixed'+str(i)+'_'+visit,z[visit]['raw_activation'],3,True)
        assert z['pass1']['raw_gate']<0<z['pass3']['raw_gate']
        assert z['pass1']['raw_up']*z['pass3']['raw_up']>0
    mixed=r['mixed_activity_case']['examples']
    assert mixed[0]['pass1']['raw_up']>0 and mixed[1]['pass1']['raw_up']<0
    name=r['number_name_case'];assert all(name['expert105_selected_per_pass'])
    assert name['next_target_position']==name['input_position']+1
    number('next_digit_native_probability',name['native_probability']*100,2)
    number('next_digit_deleted_probability',name['deleted_probability']*100,2)
    number('next_digit_loss_decrease',-name['loss_change'],5)
    for method,estimate in r['fixed_test']['mae'].items():number('forecast_mae_'+method,100*estimate['mean'],5)
    for unit in r['fixed_test']['per_unit']:
        if unit['expert'] not in [52,83,44] or not unit['is_primary_replication']:continue
        for estimate in unit['adjusted_effects']:
            number('controlled_effect_'+str(unit['expert'])+'_pass'+str(estimate['visit']),abs(100*estimate['mean']),5)
    assert sum(x['passes_sign_reversal_rule'] for x in r['fixed_test']['per_unit'] if x['is_primary_replication'])==3
    assert len(r['math_test']['primary'])==4 and not any(x['passes_all_three_preregistered_requirements'] for x in r['math_test']['primary'])
    assert r['math_test']['unique_questions']==128 and r['math_test']['orders_per_question']==3
    assert len(re.findall(r'^## Example [1-5] ',article,re.M))==5
    local=[]
    for link in re.findall(r'\]\(([^)]+)\)',article):
        if link.startswith('/'):
            assert Path(link).is_file(),link
            local.append(link)
    out={'status':'passed','source_files_verified':len(r['source_sha256']),'displayed_numbers_verified':checked,
         'native_factor_equations_consistent_with_bf16_rounding':True,'autoregressive_target_offset_verified':True,
         'mixed_example_has_same_gate_flip_and_opposite_up_signs':True,
         'local_links_verified':local,'article_sha256':hashlib.sha256(a.article.read_bytes()).hexdigest(),
         'example_data_sha256':hashlib.sha256(a.examples.read_bytes()).hexdigest(),
         'scope':'Presentation audit of existing measurements. No new experiment or hypothesis test.'}
    Path('evidence/expert_blog_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['status','source_files_verified','native_factor_equations_consistent_with_bf16_rounding','autoregressive_target_offset_verified']}))


if __name__=='__main__':main()
