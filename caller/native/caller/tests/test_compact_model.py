import os,unittest
import numpy as np
import torch
from model import NeuralCaller,ModelConfig,predict,tensors
from test_model import example

class CompactInferenceTests(unittest.TestCase):
    def test_local_sequence_reuse_with_padding_and_multiple_edits(self):
        torch.manual_seed(732);m=NeuralCaller(ModelConfig(hidden=16,dropout=0)).eval()
        seq=torch.zeros((9,12,256),dtype=torch.long);am=torch.zeros((9,12),dtype=torch.bool)
        for r in range(9):
            seq[r,:4]=torch.randint(1,6,(1,256));am[r,:4]=True
            for a,sites in enumerate(((64,),(70,82),(0,255)),1):
                for site in sites:seq[r,a,site]=(seq[r,a,site]%5)+1
            if r%3==0:seq[r,:,230:]=0
            if r%3==1:seq[r,2,229:]=0
            if r%3==2:seq[r,1,10:36]=torch.randint(1,6,(26,))
        with torch.inference_mode():
            expected=m.sequence(m.bases(seq[am]).transpose(1,2)).transpose(1,2)
            actual=m.compact_sequence_encoding(seq,am)
        torch.testing.assert_close(actual,expected,atol=3e-7,rtol=2e-6)

    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def check(self,device):
        if str(device).startswith('cuda'):
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.manual_seed(165);m=NeuralCaller(ModelConfig(hidden=16,dropout=0)).to(device).eval()
        torch.nn.init.normal_(m.pair_joint[-1].weight,std=.1)
        for populated in (False,True):
            x=example()
            if populated:x['context_strength'][:]=.7;x['cohort_context'][:]=.4
            x['allele_mask'][-1]=True;x['genotype_mask'][-1]=True
            x['base_gp'][-1]=1/78;x['read_pl'][-1]=20
            expected=predict(m,x);aux=m(tensors(x,device),return_aux=True)
            m.compact_inference=True
            actual=predict(m,x);got=m(tensors(x,device),return_aux=True)
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=2e-6)
            np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1))
            self.assertTrue(np.all(actual[~x['genotype_mask']]==0))
            for key in ('logits','read_logits','cohort_gate'):
                torch.testing.assert_close(got[key],aux[key],atol=2e-6,rtol=2e-6)
            m.compact_inference=False
    def test_cpu_nonzero_residual_masks_empty_fragments_and_cohort(self):self.check('cpu')
    @unittest.skipUnless(os.environ.get('VARDEPTH_TEST_CUDA')=='1','Real CUDA required')
    def test_cuda_nonzero_residual_masks_empty_fragments_and_cohort(self):self.check('cuda')
    def test_training_unchanged(self):
        torch.manual_seed(17);m=NeuralCaller(ModelConfig(hidden=16,dropout=0,cohort_dropout=0,prior_dropout=0)).train()
        torch.nn.init.normal_(m.pair_joint[-1].weight,std=.1);x=tensors(example(),'cpu')
        baseline=m(x,return_aux=True);m.compact_inference=True;actual=m(x,return_aux=True)
        for key in baseline:torch.testing.assert_close(actual[key],baseline[key],atol=0,rtol=0)

    def test_compact_genotype_width_with_nonzero_trained_layers(self):
        torch.manual_seed(993);m=NeuralCaller(ModelConfig(hidden=16,dropout=0)).eval()
        torch.nn.init.normal_(m.pair_joint[-1].weight,std=.1)
        for n in (1,2,6):
            x=example(n)
            for active in (False,True):
                if active:x['context_strength'][:]=.6;x['cohort_context'][:]=.3
                m.compact_inference=False;expected=predict(m,x)
                m.compact_inference=True;actual=predict(m,x)
                np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=2e-6)
                np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1))
                self.assertTrue(np.all(actual[~x['genotype_mask']]==0))

if __name__=='__main__':unittest.main()
