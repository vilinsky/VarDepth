"""Independent probabilistic check and regression cases for new inference paths."""
import argparse,ctypes,json,pathlib,sys,tempfile,unittest
import numpy as np
import pysam
HERE=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from scaffold import impute
from indel_refine import observations,repeat_bounds,kernel,owns_haplotype
from fuse import run as fuse

def exact_diploid(h,gl,r,mu):
    m,k=h.shape;x=mu+(1-2*mu)*h
    a=x[:,:,None];b=x[:,None,:]
    emission=np.stack(((1-a)*(1-b),a*(1-b)+(1-a)*b,a*b),axis=-1).reshape(m,k*k,3)
    joint=emission*np.exp(gl-gl.max(1,keepdims=True))[:,None,:];total=joint.sum(2)
    fw=np.empty((m,k*k));bk=np.ones_like(fw);fw[0]=total[0]/total[0].sum();trans=[]
    for t in range(1,m):
        one=(1-r[t])*np.eye(k)+r[t]/k;transition=np.kron(one,one);trans.append(transition)
        fw[t]=(fw[t-1]@transition)*total[t];fw[t]/=fw[t].sum()
    for t in range(m-2,-1,-1):
        bk[t]=trans[t]@(total[t+1]*bk[t+1]);bk[t]/=bk[t].max()
    state=fw*bk;state/=state.sum(1,keepdims=True)
    return np.sum(state[:,:,None]*joint/total[:,:,None],axis=1)

class ScaffoldTests(unittest.TestCase):
    def test_sampling_matches_exact_diploid_posterior(self):
        h=np.array([[0,0,1,1],[0,1,0,1],[1,1,0,0],[0,1,1,0]],float)
        gl=np.array([[0,-1,-3],[-2,0,-2],[-3,-.3,0],[0,0,0]],float);r=np.array([0,.13,.3,.19])
        exact=exact_diploid(h,gl,r,.01)
        actual=impute(h,gl,r,mutation=.01,burn=100,iterations=8000,chains=2,seed=41)
        np.testing.assert_allclose(actual,exact,atol=.025,rtol=0)
        repeated=impute(h,gl,r,mutation=.01,burn=100,iterations=8000,chains=2,seed=41)
        np.testing.assert_array_equal(actual,repeated)

    def test_unobserved_snps_and_invalid_probabilities(self):
        h=np.tile([0.,0.,1.,1.],(5,1));gl=np.zeros((5,3));r=np.full(5,.2)
        gp=impute(h,gl,r,burn=100,iterations=4000,seed=13)
        np.testing.assert_allclose(gp,np.tile([.25,.5,.25],(5,1)),atol=.035,rtol=0)
        with self.assertRaises(ValueError):impute(h,gl,r,iterations=0)
        with self.assertRaises(ValueError):impute(h*2,gl,r)

class RepeatAnchorsTests(unittest.TestCase):
    def test_shifted_insertion_has_identical_evidence_and_cram_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);seq='GCTG'+'A'*12+'CGTCGTGCTGCGTACGCTAGC'
            fa=root/'ref.fa';fa.write_text('>chr1\n'+seq+'\n');pysam.faidx(str(fa))
            header={'HD':{'VN':'1.6','SO':'coordinate'},'SQ':[{'SN':'chr1','LN':len(seq)}],
                    'RG':[{'ID':'rg','SM':'S','LB':'L','PL':'ILLUMINA'}]}
            bam=root/'a.bam';query=seq[:4]+'AA'+seq[4:]
            # A flank SNP must not eliminate the entire alternate haplotype.
            query=query[:2]+'C'+query[3:]
            with pysam.AlignmentFile(str(bam),'wb',header=header) as out:
                for i,p in enumerate([4,16]):
                    r=pysam.AlignedSegment(out.header);r.query_name='molecule'+str(i);r.query_sequence=query
                    r.query_qualities=pysam.qualitystring_to_array('I'*len(query));r.reference_id=0;r.reference_start=0
                    r.cigarstring=f'{p}M2I{len(seq)-p}M';r.mapping_quality=60;r.set_tag('RG','rg');out.write(r)
            pysam.index(str(bam));fetch=lambda a,b:seq[max(a,0):b]
            record={'start0':3,'end0':4,'evidence_start0':3,'evidence_end0':4,'alleles':['G','GAA']}
            start,end=repeat_bounds(record,fetch);self.assertGreaterEqual(end,17)
            obs=observations([str(bam)],'chr1',start,end,fetch,reference=str(fa))
            self.assertEqual(len(obs),2);self.assertEqual(obs[0]['inner'],obs[1]['inner'])
            self.assertEqual(len(obs[0]['inner']),end-start+2)
            # Same molecules in another BAM must not inflate support.
            self.assertEqual(len(observations([str(bam),str(bam)],'chr1',start,end,fetch)),2)
            cram=root/'a.cram'
            with pysam.AlignmentFile(str(bam)) as inp,pysam.AlignmentFile(str(cram),'wc',header=inp.header,reference_filename=str(fa)) as out:
                for r in inp:out.write(r)
            pysam.index(str(cram));other=observations([str(cram)],'chr1',start,end,fetch,reference=str(fa))
            self.assertEqual(obs,other)

    def test_proposals_cannot_duplicate_an_event_in_the_neighbor_block(self):
        seq='CGT'+'A'*10+'CGTGC';fetch=lambda a,b:seq[max(0,a):b]
        record={'start0':8,'end0':16}
        self.assertFalse(owns_haplotype(record,2,seq[2:16],seq[2:10]+'A'+seq[10:16],fetch))
        self.assertTrue(owns_haplotype(record,2,seq[2:16],seq[2:16],fetch))

    def test_global_likelihood_prefers_correct_insertion_and_deletion(self):
        lib=kernel()
        for read,wrong in [('ACGTTACG','ACGTACG'),('ACGTACG','ACGTTACG')]:
            q=(ctypes.c_ubyte*len(read))(*([40]*len(read)))
            score=lambda hap:lib.anchored_pairhmm(read.encode(),q,len(read),hap.encode(),len(hap),.001,.1)
            self.assertGreater(score(read),score(wrong)+3.)

class FusionTests(unittest.TestCase):
    def test_single_sample_expanded_genotypes_and_likelihood_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);old=root/'old.vcf.gz';h=pysam.VariantHeader();h.contigs.add('chr1',length=1000);h.add_sample('S')
            for name,n,t in [('GT',1,'String'),('DP',1,'Integer'),('GP','G','Float'),('BGP','G','Float'),('PL','G','Integer')]:h.formats.add(name,n,t,name)
            with pysam.VariantFile(str(old),'wz',header=h) as out:
                r=out.new_record(contig='chr1',start=10,alleles=['A','AT']);c=r.samples['S'];c['GT']=(0,1);c['DP']=100
                c['GP']=(.01,.98,.01);c['BGP']=c['GP'];c['PL']=(30,0,30);out.write(r)
            alleles=['A','AT']+['A'+'T'*i for i in range(2,13)];n=len(alleles);gp=np.zeros(n*(n+1)//2);gp[-1]=1
            ref=root/'ref.json';ref.write_text(json.dumps([dict(record_id='chr1:11:A:AT',start0=10,alleles=alleles,added_alleles=n-2,samples={'S':dict(trusted=True,gp=gp.tolist(),pl=[0]*len(gp),dp=60,exact_ad=[0]*(n-1)+[60])})]))
            dest=root/'new.vcf.gz';fuse(argparse.Namespace(neural=str(old),scaffold=None,refinements=str(ref),config=None,output=str(dest)))
            with pysam.VariantFile(str(dest)) as inp:
                r=next(inp);c=r.samples['S'];self.assertEqual(c['GT'],(n-1,n-1));self.assertEqual(len(c['PL']),len(gp))
                self.assertAlmostEqual(sum(c['GP']),1,places=6);self.assertEqual(len(c['NGP']),len(gp));self.assertNotIn('AD',c)

if __name__=='__main__':unittest.main()
