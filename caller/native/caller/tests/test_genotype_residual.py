import unittest,numpy as np,torch,pysam
from trust import bundle,residual_model_class,MAX_GP
class GenotypeResidualTests(unittest.TestCase):
 def record(self):
  h=pysam.VariantHeader();h.contigs.add('chrToy',length=100);h.formats.add('GT',1,'String','GT');h.formats.add('GP','G','Float','GP');h.formats.add('DP',1,'Integer','DP');h.add_sample('S')
  r=h.new_record(contig='chrToy',start=10,alleles=['A','AT']);r.samples['S']['GT']=(0,1);r.samples['S']['GP']=(.1,.8,.1);r.samples['S']['DP']=10;return r
 def test_ambiguous_observations_do_not_create_balance(self):
  r=self.record();ex=dict(dp=10,gp=[.1,.8,.1],pl=[20,0,20],exact_ad=[0,0],exact_genotype_counts=[[10,0,0],[0,0,10],[10,0,0]],trusted=False,support_fraction=1.)
  row=dict(start0=10,end0=11,alleles=['A','AT'],added_alleles=0,samples={'S':ex});item=bundle(r,'S',row)
  f=item['genotype_features'];self.assertEqual(f.shape,(MAX_GP,20));self.assertEqual(f[1,5],1);self.assertEqual(f[1,6],0);self.assertEqual(f[1,7],0);self.assertEqual(f[1,8],1)
 def test_frozen_snv_and_valid_padding(self):
  torch.manual_seed(11);model=residual_model_class()();model.eval();x=torch.rand(2,10,42);x[0,:,2]=0;x[1,:,2]=1
  p=torch.rand(2,10,6);p[0,:,3:]=0;p/=p.sum(2,keepdim=True);available=torch.ones(2,10,dtype=torch.bool);n=torch.tensor([3,6]);gf=torch.zeros(2,6,20)
  with torch.inference_mode():a,w=model(x,p,available,n,gf);b,bw=model.parent(x,p,available,n)
  self.assertTrue(torch.equal(a[0],b[0]));self.assertTrue(torch.equal(w,bw));self.assertTrue(torch.equal(a[0,3:],torch.zeros(3)));self.assertTrue(torch.allclose(a.sum(1),torch.ones(2)))
  self.assertTrue(all(not q.requires_grad for q in model.parent.parameters()))
if __name__=='__main__':unittest.main()
