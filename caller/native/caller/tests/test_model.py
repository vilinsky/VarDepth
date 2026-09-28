import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import NeuralCaller,ModelConfig,predict,tensors,load_model,MODEL_VERSION

def example(n=6):
    rng=np.random.default_rng(41);a=np.array([2,3,4,2,2,3][:n]);g=a*(a+1)//2
    am=np.arange(12)[None,:]<a[:,None];gm=np.arange(78)[None,:]<g[:,None]
    gp=rng.random((n,78))*gm;gp/=gp.sum(1,keepdims=True)
    return {'allele_sequences':rng.integers(1,5,(n,12,256),dtype=np.int8)*am[:,:,None],
        'allele_mask':am,'genotype_mask':gm,'fragment_features':rng.normal(size=(n,64,20)).astype('f'),
        'fragment_mask':np.arange(64)[None,:]<np.array([0,3,20,60,1,8][:n])[:,None],
        'base_gp':gp.astype('f'),'read_pl':(-10*np.log10(np.maximum(gp,1e-30))).astype('f'),
        'depth':np.array([0,3,20,100,1,8][:n],dtype='f'),'allele_features':rng.random((n,12,8)).astype('f'),
        'cohort_context':np.zeros((n,12,6),dtype='f'),'context_strength':np.zeros(n,dtype='f')}

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_empty_molecules_and_multiallelic_mask(self):
        x=example();m=NeuralCaller().eval();p=predict(m,x)
        self.assertTrue(np.isfinite(p).all());np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
        self.assertTrue((p[~x['genotype_mask']]==0).all())
        np.testing.assert_array_equal(p.argmax(1),x['base_gp'].argmax(1))
    def test_single_person_equals_explicit_empty_context(self):
        x=example();m=NeuralCaller().eval()
        # Make the correction active so zero-initialized output cannot hide a
        # dependence on a supposedly absent cohort.
        torch.nn.init.normal_(m.pair_joint[-1].weight,std=.1)
        a=predict(m,x);del x['cohort_context'];del x['context_strength']
        np.testing.assert_array_equal(a,predict(m,x))
    def test_molecule_permutation(self):
        x=example();m=NeuralCaller().eval();torch.nn.init.normal_(m.pair_joint[-1].weight,std=.1)
        a=predict(m,x);order=np.arange(64)[::-1].copy()
        x['fragment_features']=x['fragment_features'][:,order];x['fragment_mask']=x['fragment_mask'][:,order]
        np.testing.assert_allclose(a,predict(m,x),atol=2e-7,rtol=2e-6)
    def test_cohort_is_connected_and_zero_strength_blocks_it(self):
        x=example();m=NeuralCaller().eval();torch.nn.init.normal_(m.pair_joint[-1].weight,std=.1)
        a=predict(m,x);x['cohort_context'][:]=.7
        np.testing.assert_array_equal(a,predict(m,x))
        x['context_strength'][:]=1
        self.assertGreater(float(np.max(np.abs(a-predict(m,x)))),1e-7)
    def test_supervised_gradient_can_improve_loss(self):
        torch.manual_seed(9);x=example();m=NeuralCaller(ModelConfig(hidden=16,dropout=0,cohort_dropout=0,prior_dropout=0))
        b=tensors(x,'cpu');labels=torch.tensor([1,2,1,2,1,2]);optimizer=torch.optim.Adam(m.parameters(),lr=.01)
        initial=float(torch.nn.functional.cross_entropy(m(b),labels))
        for _ in range(15):
            optimizer.zero_grad();z=m(b,return_aux=True)
            loss=torch.nn.functional.cross_entropy(z['logits'],labels)+.1*torch.nn.functional.cross_entropy(z['read_logits'],labels)
            loss.backward();optimizer.step()
        self.assertLess(float(torch.nn.functional.cross_entropy(m(b),labels)),initial*.8)
    def test_checkpoint_roundtrip(self):
        m=NeuralCaller().eval();x=example()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.pt';torch.save({'model_version':MODEL_VERSION,'config':asdict(m.config),'model_state':m.state_dict(),'temperature':1.},p)
            np.testing.assert_array_equal(predict(m,x),predict(load_model(p),x))
    def test_wrong_certain_external_anchor_can_be_overridden(self):
        torch.manual_seed(193)
        x=example(1);x['base_gp'][:]=0;x['base_gp'][0,0]=1
        x['read_pl'][0,:3]=[100,0,100];x['depth'][0]=30
        x['allele_features'][0,:2,0]=.5
        model=NeuralCaller(ModelConfig(hidden=16,dropout=0,cohort_dropout=0,prior_dropout=0))
        b=tensors(x,'cpu');label=torch.tensor([1])
        optimizer=torch.optim.Adam(model.parameters(),lr=.005)
        for _ in range(60):
            optimizer.zero_grad()
            loss=torch.nn.functional.cross_entropy(model(b),label)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        self.assertEqual(int(model(b).argmax(1)),1)
    def test_read_auxiliary_has_no_direct_population_inputs(self):
        x=example();m=NeuralCaller().eval()
        a=m(tensors(x,'cpu'),return_aux=True)['read_logits'].detach().numpy()
        x['base_gp']=np.roll(x['base_gp'],1,axis=1)*x['genotype_mask']
        x['base_gp']/=x['base_gp'].sum(1,keepdims=True)
        x['allele_features'][:,:,1]=.99;x['allele_features'][:,:,5]=1
        x['cohort_context'][:]=.8;x['context_strength'][:]=1
        b=m(tensors(x,'cpu'),return_aux=True)['read_logits'].detach().numpy()
        np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()
