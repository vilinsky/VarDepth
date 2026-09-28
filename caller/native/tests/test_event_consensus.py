import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import pysam
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'caller'))
from event_ownership import reconcile,project_gp,same_phase
from ownership_events import catalog_events

class EventConsensusTests(unittest.TestCase):
    def fixture(self,directory,confidence=(.8,.999),gts=((0,1),(0,1))):
        ref='ACGTGCAATGCTACGATCGTACCGTAGCATGACTGATCGATCGTACGTAGCTAGCTACGTA'
        path=directory/'ref.fa';path.write_text('>chr1\n'+ref+'\n');pysam.faidx(str(path))
        header=pysam.VariantHeader();header.contigs.add('chr1',length=len(ref));header.add_sample('S')
        for field,number,kind in [('GT',1,'String'),('GP','G','Float'),('GQ',1,'Integer'),('DP',1,'Integer'),('CS',1,'Float'),('DS','A','Float')]:
            header.formats.add(field,number,kind,field)
        header.filters.add('LowQual',None,None,'low')
        # The long, lower-confidence record contains a deletion plus a unique SNP.
        original=ref[10:30];altered=original[:5]+original[6:]
        snp_index=14;replacement=next(x for x in 'ACGT' if x!=altered[snp_index])
        altered=altered[:snp_index]+replacement+altered[snp_index+1:]
        vc=directory/'in.vcf.gz'
        with pysam.VariantFile(vc,'wz',header=header) as output:
            for index,(start,alleles) in enumerate([(10,[original,altered]),(14,[ref[14:16],ref[14]])]):
                v=header.new_record(contig='chr1',start=start,alleles=alleles)
                gt=gts[index];g=max(gt)*(max(gt)+1)//2+min(gt);gp=np.full(3,(1-confidence[index])/2);gp[g]=confidence[index]
                v.samples['S']['GT']=gt;v.samples['S']['GP']=tuple(gp);v.samples['S']['DP']=40;v.samples['S']['GQ']=7
                v.filter.add('PASS');output.write(v)
        pysam.tabix_index(str(vc),preset='vcf',force=True)
        return path,vc

    def test_short_confident_source_keeps_unique_child_snp(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);fa,vc=self.fixture(d);out=d/'out.vcf.gz';result=reconcile(vc,fa,out)
            self.assertEqual(result['counts']['removed_duplicate_events'],1)
            self.assertEqual(result['proofs'][0]['owner_index'],1)
            with pysam.FastaFile(fa) as ref,pysam.VariantFile(out) as stream:
                called=[]
                for v in stream:
                    gp=v.samples['S']['GP'];self.assertAlmostEqual(sum(gp),1,places=6)
                    events=catalog_events(v.start,list(v.alleles),lambda s,e:ref.fetch('chr1',s,e).upper())
                    called.extend(e for g in set(v.samples['S']['GT']) for e in events[g])
                self.assertEqual(sum(len(r)!=len(a) for p,r,a in called),1)
                self.assertEqual(sum(len(r)==len(a)==1 for p,r,a in called),1)

    def test_majority_supported_duplicates_do_not_need_gq20(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);fa,vc=self.fixture(d,confidence=(.6,.7))
            result=reconcile(vc,fa,d/'out.vcf.gz')
            self.assertEqual(result['counts']['removed_duplicate_events'],1)

    def test_different_diploid_copy_number_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);fa,vc=self.fixture(d,gts=((1,1),(0,1)))
            result=reconcile(vc,fa,d/'out.vcf.gz')
            self.assertEqual(result['counts'].get('removed_duplicate_events',0),0)

    def test_exact_projection_and_phase_conflict(self):
        p=project_gp([.1,.2,.1,.2,.3,.1],[0,1,1])
        np.testing.assert_allclose(p,[.1,.4,.5],rtol=0,atol=1e-15)
        e1=(10,'AC','A');e2=(20,'TG','T');events=frozenset([e1,e2])
        self.assertFalse(same_phase({'S':(events,frozenset())},{'S':(frozenset([e1]),frozenset([e2]))},events))

    def cohort_fixture(self,directory,second_gts):
        fa,vc=self.fixture(directory)
        with pysam.VariantFile(vc) as source:
            rows=[v.copy() for v in source];header=source.header.copy()
        header.add_sample('SECOND');cohort=directory/'cohort.vcf.gz'
        with pysam.VariantFile(cohort,'wz',header=header) as output:
            for index,original in enumerate(rows):
                v=header.new_record(contig=original.chrom,start=original.start,alleles=original.alleles)
                for field,value in original.samples['S'].items():v.samples['S'][field]=value
                gt=second_gts[index];g=max(gt)*(max(gt)+1)//2+min(gt);p=[.0005]*3;p[g]=.999
                v.samples['SECOND']['GT']=gt;v.samples['SECOND']['GP']=p;v.samples['SECOND']['DP']=0
                v.filter.add('PASS');output.write(v)
        pysam.tabix_index(str(cohort),preset='vcf',force=True)
        return fa,cohort

    def test_cohort_disagreement_in_zero_depth_sample_protects_event(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);fa,vc=self.cohort_fixture(d,[(0,1),(0,0)])
            result=reconcile(vc,fa,d/'out.vcf.gz')
            self.assertEqual(result['counts'].get('removed_duplicate_events',0),0)

    def test_cohort_agreement_retains_zero_depth_sample_posterior(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);fa,vc=self.cohort_fixture(d,[(0,1),(0,1)]);out=d/'out.vcf.gz'
            result=reconcile(vc,fa,out)
            self.assertEqual(result['counts']['removed_duplicate_events'],1)
            with pysam.VariantFile(out) as source:
                for v in source:
                    self.assertEqual(v.samples['SECOND']['DP'],0)
                    self.assertAlmostEqual(sum(v.samples['SECOND']['GP']),1,places=6)

if __name__=='__main__':unittest.main()
