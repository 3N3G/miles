"""Analytic teacher checks for native edits, dose controls and observation order."""

import unittest
import torch
from native_expert import RouterEdit,letter_metrics,render_question
from native_expert_confirmation import OutputNormControl
from capture_native_neurons import NativeNeuronObserver


class TeacherExperts(torch.nn.Module):
    def __init__(self):
        super().__init__();self.act_fn=torch.nn.Identity()

    def forward(self,x,indices,weights):
        result=torch.zeros_like(x)
        for expert in indices.unique():
            token,slot=torch.where(indices==expert)
            y=self.act_fn(x[token]*(int(expert)+1))
            result.index_add_(0,token,y*weights[token,slot,None])
        return result


class NativeExpertChecks(unittest.TestCase):
    def setUp(self):
        self.x=torch.arange(96,dtype=torch.float32).reshape(12,8)/50
        self.indices=torch.tensor([[0,1,2]]*12)
        self.weights=torch.tensor([[.2,.3,.5]]*12)
        self.module=TeacherExperts()

    def test_exact_deletion_matches_analytic_branch(self):
        base=self.module(self.x,self.indices,self.weights)
        edit=RouterEdit(1);w=edit(None,None,(None,self.weights,self.indices))[1]
        actual=self.module(self.x,self.indices,w)
        self.assertTrue(torch.allclose(base-actual,self.x*.6,atol=1e-6))
        self.assertTrue(torch.equal(self.weights,torch.tensor([[.2,.3,.5]]*12)))

    def test_visit_and_position_controls(self):
        mask=torch.arange(12)%2==0
        edit=RouterEdit(1,visits=[2],position_mask=mask)
        original=(None,self.weights,self.indices)
        self.assertIs(edit(None,None,original),original)
        w=edit(None,None,original)[1]
        self.assertTrue(torch.equal(w[~mask],self.weights[~mask]))
        self.assertTrue((w[mask,1]==0).all())
        self.assertIs(edit(None,None,original),original)

    def test_renormalization_and_absent_expert(self):
        w=RouterEdit(1,renormalize=True)(None,None,(None,self.weights,self.indices))[1]
        self.assertTrue(torch.allclose(w.sum(-1),self.weights.sum(-1)))
        self.assertTrue((w[:,1]==0).all())
        w=RouterEdit(128)(None,None,(None,self.weights,self.indices))[1]
        self.assertTrue(torch.equal(w,self.weights))

    def test_equal_norm_noise_and_output_deletion(self):
        base=self.module(self.x,self.indices,self.weights)
        args=(self.x,self.indices,self.weights)
        deleted=OutputNormControl(1,[1],7,return_deleted=True)(self.module,args,base)
        random=OutputNormControl(1,[1],7)(self.module,args,base)
        self.assertTrue(torch.allclose((random-base).norm(dim=-1),(deleted-base).norm(dim=-1),atol=1e-6))
        self.assertFalse(torch.allclose(random,deleted))

    def test_native_unit_observation_matches_known_teacher(self):
        base=self.module(self.x,self.indices,self.weights)
        observer=NativeNeuronObserver([1]);handles=[self.module.register_forward_pre_hook(observer.begin),
            self.module.act_fn.register_forward_hook(observer.activation),self.module.register_forward_hook(observer.end)]
        try:actual=self.module(self.x,self.indices,self.weights)
        finally:
            for h in handles:h.remove()
        self.assertTrue(torch.equal(base,actual));self.assertEqual(len(observer.rows),1)
        row=observer.rows[0];self.assertEqual(row['expert'],1)
        self.assertTrue(torch.equal(torch.from_numpy(row['activation']),self.x*2))
        self.assertTrue(torch.equal(torch.from_numpy(row['weights']),self.weights[:,1]))

    def test_alias_probability_and_answer_order(self):
        metrics=letter_metrics(torch.zeros(5),{'A':[0,4],'B':[1],'C':[2],'D':[3]},'A')
        self.assertAlmostEqual(metrics['letter_mass'],1,places=6)
        self.assertAlmostEqual(metrics['choice_probabilities'][0],.4,places=6)
        text,gold=render_question({'question':'Q','choices':['a','b','c','d'],'answer':2},[2,0,3,1])
        self.assertEqual(gold,'A');self.assertIn('A. c',text)

    def test_grouped_native_observation_recovers_expert_token_mapping(self):
        class Grouped(torch.nn.Module):
            def __init__(self):super().__init__();self.act_fn=torch.nn.Identity()
            def forward(self,x,indices,weights):
                ids,perm=indices.reshape(-1).sort();token=perm//indices.shape[-1]
                h=self.act_fn(x[token]*(ids[:,None]+1))
                weighted=h*weights.reshape(-1)[perm,None]
                inverse=torch.empty_like(perm);inverse[perm]=torch.arange(len(perm))
                return weighted[inverse].reshape(*indices.shape,-1).sum(1)
        m=Grouped();x=torch.arange(12,dtype=torch.float32).reshape(3,4)
        ix=torch.tensor([[2,0],[1,2],[0,1]]);w=torch.tensor([[.2,.8],[.3,.7],[.4,.6]])
        base=m(x,ix,w);o=NativeNeuronObserver([1]);handles=[m.register_forward_pre_hook(o.begin),
            m.act_fn.register_forward_hook(o.activation),m.register_forward_hook(o.end)]
        try:observed=m(x,ix,w)
        finally:
            for h in handles:h.remove()
        self.assertTrue(torch.equal(base,observed));self.assertEqual(len(o.rows),1)
        row=o.rows[0]
        for i,pos in enumerate(row['positions']):
            self.assertTrue(torch.equal(torch.from_numpy(row['activation'][i]),x[pos]*2))
            self.assertAlmostEqual(float(row['weights'][i]),.3 if pos==1 else .6,places=6)


if __name__=='__main__':unittest.main()
