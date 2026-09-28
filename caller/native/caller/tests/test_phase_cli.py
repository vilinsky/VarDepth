import argparse,json,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
import pysam
from phase_cli import extract,components,run

class PhaseVCFContracts(unittest.TestCase):
    def fixture(self,root):
        ref=root/'ref.fa';ref.write_text('>chr1\n'+'A'*100+'\n');pysam.faidx(str(ref))
        bam=root/'reads.bam'
        header={'HD':{'VN':'1.6','SO':'coordinate'},'SQ':[{'SN':'chr1','LN':100}],
                'RG':[{'ID':'r','SM':'S','PL':'ILLUMINA','LB':'L'}]}
        with pysam.AlignmentFile(str(bam),'wb',header=header) as out:
            for k in range(6):
                seq=list('A'*30);seq[10 if k%2==0 else 15]='C' if k%2==0 else 'G'
                read=pysam.AlignedSegment(out.header);read.query_name='m'+str(k//2)+'-'+str(k%2)
                read.query_sequence=''.join(seq);read.query_qualities=[30]*30;read.flag=0
                read.reference_id=0;read.reference_start=0;read.mapping_quality=60;read.cigar=[(0,30)];read.set_tag('RG','r')
                out.write(read);out.write(read) # Same molecule encountered twice.
        pysam.index(str(bam));vcf=root/'calls.vcf.gz';h=pysam.VariantHeader();h.contigs.add('chr1',length=100)
        h.add_sample('S');h.formats.add('GT',1,'String','GT');h.formats.add('GQ',1,'Integer','GQ')
        h.formats.add('GP','G','Float','GP');h.formats.add('DS','A','Float','DS')
        with pysam.VariantFile(str(vcf),'wz',header=h) as out:
            for pos,alt in [(10,'C'),(15,'G')]:
                r=out.new_record(contig='chr1',start=pos,alleles=('A',alt));r.samples['S']['GT']=(0,1)
                r.samples['S']['GQ']=50;r.samples['S']['GP']=(.01,.98,.01);r.samples['S']['DS']=(1.,);out.write(r)
        pysam.tabix_index(str(vcf),preset='vcf',force=True);return ref,bam,vcf
    def test_molecules_and_phase_only_preservation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ref,bam,vcf=self.fixture(root)
            sites,blocks,stats=extract(str(vcf),[str(bam)],str(ref),'S','chr1:1-50')
            self.assertEqual(stats['molecules_with_links'],6);self.assertEqual(len(blocks),1)
            args=argparse.Namespace(vcf=str(vcf),bam=[str(bam)],reference=str(ref),sample='S',region='chr1:1-50',
                max_qubits=12,min_gq=20,output=str(root/'phase'),phased_vcf=str(root/'phased.vcf.gz'),backend='exact',
                max_blocks=16,seed=17,layers=2,shots=256,max_evaluations=12)
            run(args)
            with pysam.VariantFile(str(vcf)) as a,pysam.VariantFile(args.phased_vcf) as b:
                old=list(a);new=list(b)
                self.assertEqual([r.samples['S']['GT'] for r in new],[(0,1),(1,0)])
                for left,right in zip(old,new):
                    for field in ['GP','DS','GQ']:self.assertEqual(left.samples['S'][field],right.samples['S'][field])
                    self.assertTrue(right.samples['S'].phased);self.assertEqual(right.samples['S']['PS'],11)
            args.backend='exact-elimination';args.output=str(root/'elimination');args.phased_vcf=str(root/'elimination.vcf.gz')
            run(args)
            with pysam.VariantFile(args.phased_vcf) as f:self.assertEqual([str(r) for r in f],[str(r) for r in new])
            for reason,answer in [('unsolved',dict(solved=False,reason='budget')),
                                  ('tied',dict(solved=True,bits=[0,0],nll=1.,next_solution_gap=0.))]:
                args.output=str(root/reason);args.phased_vcf=str(root/(reason+'.vcf.gz'))
                with patch('phase_search.eliminate_unique',return_value=answer):run(args)
                with pysam.VariantFile(args.phased_vcf) as f:
                    self.assertTrue(all(not r.samples['S'].phased for r in f))
    def test_splitting_does_not_phase_disconnected_sites(self):
        self.assertEqual(components([0,1,2,3],[(0,2),(1,3)]),[[0,2],[1,3]])

    def test_wrong_bam_sample_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ref,bam,vcf=self.fixture(root);bad=root/'wrong.bam'
            with pysam.AlignmentFile(str(bam)) as inp:
                h=inp.header.to_dict();h['RG'][0]['SM']='OTHER'
                with pysam.AlignmentFile(str(bad),'wb',header=h) as out:
                    for r in inp:out.write(r)
            pysam.index(str(bad))
            with self.assertRaisesRegex(ValueError,'selected VCF sample'):
                extract(str(vcf),[str(bad)],str(ref),'S','chr1:1-50')

    def test_existing_phase_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ref,bam,vcf=self.fixture(root);phased=root/'existing.vcf.gz'
            with pysam.VariantFile(str(vcf)) as inp,pysam.VariantFile(str(phased),'wz',header=inp.header) as out:
                for r in inp:r.samples['S'].phased=True;out.write(r)
            pysam.tabix_index(str(phased),preset='vcf',force=True)
            sites,blocks,stats=extract(str(phased),[str(bam)],str(ref),'S','chr1:1-50')
            self.assertEqual(sites,[]);self.assertEqual(blocks,[])
            self.assertEqual(stats['existing_phased_snvs_skipped'],2)

if __name__=='__main__':unittest.main()
