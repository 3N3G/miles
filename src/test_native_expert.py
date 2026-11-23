"""Analytic teacher checks for native edits, dose controls and observation order."""

import unittest
import torch
from native_expert import RouterEdit,letter_metrics,render_question
from native_expert_confirmation import OutputNormControl
from capture_native_neurons import NativeNeuronObserver
from native_unit_routing import UnitControl,ReaderObserver


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

    def test_native_grouped_unit_deletion_and_norm_control(self):
        class Grouped(torch.nn.Module):
            def __init__(self):super().__init__();self.act_fn=torch.nn.Identity()
            def forward(self,x,indices,weights):
                ids,perm=indices.reshape(-1).sort();token=perm//indices.shape[-1]
                h=self.act_fn(x[token]*(ids[:,None]+1));weighted=h*weights.reshape(-1)[perm,None]
                inverse=torch.empty_like(perm);inverse[perm]=torch.arange(len(perm))
                return weighted[inverse].reshape(*indices.shape,-1).sum(1)
        m=Grouped();x=torch.arange(12,dtype=torch.float32).reshape(3,4)
        ix=torch.tensor([[2,0],[1,2],[0,1]]);w=torch.tensor([[.2,.8],[.3,.7],[.4,.6]])
        base=m(x,ix,w)
        def evaluate(control):
            handles=[m.register_forward_pre_hook(control.begin),m.act_fn.register_forward_hook(control.activation),m.register_forward_hook(control.end)]
            try:return m(x,ix,w)
            finally:
                for h in handles:h.remove()
        deleted=evaluate(UnitControl([(1,1)],visits=[1]))
        direct=evaluate(UnitControl([(1,1)],visits=[1],direct=True))
        self.assertTrue(torch.equal(deleted,direct))
        expected=base.clone();expected[1,1]-=x[1,1]*2*.3;expected[2,1]-=x[2,1]*2*.6
        self.assertTrue(torch.allclose(deleted,expected,atol=1e-6))
        random=evaluate(UnitControl([(1,1)],visits=[1],random_seed=7))
        self.assertTrue(torch.allclose((random-base).norm(dim=-1),(deleted-base).norm(dim=-1),atol=1e-6))

    def test_native_block_reader_reconstruction(self):
        class Reader(torch.nn.Module):
            def __init__(self):super().__init__();self.norm=torch.nn.Identity();self.query=torch.nn.Parameter(torch.tensor([1.,0.,-.5,2.]))
            def forward(self,blocks,partial):
                keys=torch.cat([blocks,partial.unsqueeze(-2)],-2)
                q=self.query.view(1,1,1,4).expand(1,3,1,4)
                weights=(q@keys.transpose(2,3)*.5).softmax(-1)
                return ((weights@keys)[:,:,0],)
        torch.manual_seed(7);m=Reader();blocks=torch.randn(1,3,2,4);partial=torch.randn(1,3,4)
        o=ReaderObserver();h=m.register_forward_hook(o)
        try:m(blocks,partial)
        finally:h.remove()
        self.assertEqual(o.errors,[0]);self.assertTrue(torch.allclose(torch.from_numpy(o.rows[0]).sum(-1),torch.ones(3)))

    def test_reader_normalization_can_reverse_writer_direction(self):
        # A writer points towards the query, but even more strongly away from it.
        # Its radial component changes normalization enough to reverse its effect.
        base=torch.tensor([1.,1.],dtype=torch.float64)
        writer=torch.tensor([1.,3.],dtype=torch.float64)
        h=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
        block=base+h*writer
        rms=block.square().mean().sqrt()
        full=torch.stack([h*0,block[0]/rms]).softmax(0)[-1]
        direct=torch.stack([h*0,block[0]/rms.detach()]).softmax(0)[-1]
        grad,=torch.autograd.grad(full,h,retain_graph=True)
        direct_grad,=torch.autograd.grad(direct,h)
        epsilon=1e-5
        changed=base-epsilon*writer
        exact=torch.stack([changed[0]*0,changed[0]/changed.square().mean().sqrt()]).softmax(0)[-1]-full.detach()
        self.assertGreater(float(exact),0)
        self.assertGreater(float(-epsilon*grad),0)
        self.assertLess(float(-epsilon*direct_grad),0)
        self.assertAlmostEqual(float(exact),float(-epsilon*grad),places=9)

    def test_native_unit_atlas_recovers_known_radial_effect(self):
        from types import SimpleNamespace
        from native_unit_atlas import sensitivity,local_delta
        class Experts(torch.nn.Module):
            def __init__(self):super().__init__();self.act_fn=torch.nn.Identity()
            def forward(self,x,indices,weights):
                groups,perm=indices.reshape(-1).sort();h=self.act_fn(x[perm])
                out=h[:,1127,None]*torch.tensor([1.,3.])
                inverse=torch.empty_like(perm);inverse[perm]=torch.arange(len(perm))
                return out[inverse]*weights
        class Norm(torch.nn.Module):
            eps=1e-6
            def forward(self,x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+self.eps)
        expert=Experts();x=torch.zeros(32,1152);x[:,1127]=.01
        indices=torch.full((32,1),45);weights=torch.ones(32,1);args=(x,indices,weights)
        y=expert(*args);pre=torch.ones(1,32,2);block=pre+y.reshape_as(pre)
        reader=SimpleNamespace(norm=Norm(),query=torch.tensor([1.,0.]))
        layer=SimpleNamespace(mlp=SimpleNamespace(experts=expert),post_mlp_convolution=torch.nn.Identity())
        capture=SimpleNamespace(expert_args=[args]*3,expert_out=[y]*3,pre=[pre]*3,attn=[torch.zeros_like(pre)]*3,
                                reader=[(torch.zeros(1,32,1,2),block)]*3,conv_kwargs=[{}]*3)
        full,dot=sensitivity(layer,reader,capture,1)
        actual=local_delta(layer,reader,capture,1,[(45,1127)])
        self.assertGreater(actual,0);self.assertGreater(full[45,1127],0);self.assertLess(dot[45,1127],0)
        # Check the gradient against the analytic derivative, not a finite deletion.
        # The latter includes curvature, which the forecast test measures separately.
        b=torch.tensor([1.01,1.03],dtype=torch.float64)
        rms=(b.square().mean()+1e-6).sqrt();s=b[0]/rms/2**.5
        ds=(1/rms-b[0]*(b@torch.tensor([1.,3.],dtype=torch.float64))/2/rms**3)/2**.5
        analytic=float(-.01*torch.sigmoid(s)*(1-torch.sigmoid(s))*ds)
        self.assertAlmostEqual(float(full[45,1127]),analytic,places=7)
        self.assertLess(abs(float(full[45,1127])-actual),abs(float(dot[45,1127])-actual))
        self.assertEqual(int((full!=0).sum()),1)


if __name__=='__main__':unittest.main()
