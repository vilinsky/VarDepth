"""Actual synthetic BAM/CRAM end-to-end feature extraction, without truth API."""
import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest
import numpy as np
import pysam

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from prepare import add_inputs,prepare
from call import load_feature_directory,emit_vcf


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        rng=random.Random(41);self.sequence=''.join(rng.choice('ACGT') for _ in range(1000))
        self.ref=self.root/'ref.fa';self.ref.write_text('>chr1\n'+self.sequence+'\n');pysam.faidx(str(self.ref))

    def tearDown(self):self.temp.cleanup()

    def bam(self,name='S',empty=False,platform='ILLUMINA'):
        path=self.root/(name+'.bam')
        header={'HD':{'VN':'1.6','SO':'coordinate'},'SQ':[{'SN':'chr1','LN':1000}],
                'RG':[{'ID':'rg','SM':name,'LB':'synthetic-library','PL':platform}]}
        with pysam.AlignmentFile(str(path),'wb',header=header) as out:
            if not empty:
                choices=[x for x in 'ACGT' if x!=self.sequence[150]][:2]
                for i in range(20):
                    r=pysam.AlignedSegment(out.header);r.query_name='q'+str(i);r.reference_id=0;r.reference_start=100
                    seq=list(self.sequence[100:200]);seq[50]=choices[i%2]
                    r.query_sequence=''.join(seq);r.query_qualities=pysam.qualitystring_to_array('I'*100)
                    r.mapping_quality=60;r.cigarstring='100M';r.flag=16 if i%3==0 else 0
                    r.set_tag('RG','rg');r.set_tag('NM',1);out.write(r)
        pysam.index(str(path));return path

    def run_prepare(self,bam,out):
        p=argparse.ArgumentParser();add_inputs(p);p.add_argument('--out');p.add_argument('--no-cohort',action='store_true')
        args=p.parse_args(['--bam','S='+str(bam),'--reference',str(self.ref),'--region','chr1:101-300','--panel-reader','python','--out',str(out)])
        prepare(args);return load_feature_directory(out)

    def test_bam_cram_exact_features_multiallelic(self):
        bam=self.bam();cram=self.root/'S.cram'
        with pysam.AlignmentFile(str(bam),'rb') as source,pysam.AlignmentFile(str(cram),'wc',header=source.header,reference_filename=str(self.ref)) as dest:
            for r in source:dest.write(r)
        pysam.index(str(cram))
        a,ma=self.run_prepare(bam,self.root/'from-bam');b,mb=self.run_prepare(cram,self.root/'from-cram')
        self.assertEqual(ma['records'],mb['records']);self.assertEqual(len(ma['records']),1)
        self.assertEqual(len(ma['records'][0]['alleles']),3)
        for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        self.assertEqual(int(a['fragment_mask'].sum()),20)
        self.assertTrue(np.any(a['fragment_features'][0,:20,14]>0))
        self.assertEqual(a['genotype_mask'][0].sum(),6)
        # The decoder must preserve a true alternate/alternate genotype.
        gp=np.zeros((1,78));gp[0,4]=1
        output=self.root/'called.vcf.gz';emit_vcf(output,a,ma,gp)
        with pysam.VariantFile(str(output)) as f:
            rec=next(f);self.assertEqual(rec.samples['S']['GT'],(1,2))
            self.assertEqual(len(rec.samples['S']['PL']),6)

    def test_zero_reads_empty_valid_vcf(self):
        arrays,metadata=self.run_prepare(self.bam(empty=True),self.root/'empty')
        self.assertEqual(len(metadata['records']),0)
        output=self.root/'empty.vcf.gz';emit_vcf(output,arrays,metadata,np.zeros((0,78)))
        with pysam.VariantFile(str(output)) as f:self.assertEqual(list(f),[])

    def test_known_non_illumina_rejected(self):
        with self.assertRaisesRegex(ValueError,'Illumina'):
            self.run_prepare(self.bam(platform='ONT'),self.root/'unsupported')

    def test_two_people_share_alleles_and_prepare_cohort_automatically(self):
        a=self.bam(name='S');b=self.bam(name='T');out=self.root/'two-people'
        p=argparse.ArgumentParser();add_inputs(p);p.add_argument('--out');p.add_argument('--no-cohort',action='store_true')
        args=p.parse_args(['--bam','S='+str(a),'--bam','T='+str(b),'--reference',str(self.ref),
                           '--region','chr1:101-300','--panel-reader','python','--out',str(out)])
        prepare(args);arrays,metadata=load_feature_directory(out)
        self.assertEqual(metadata['samples'],['S','T']);self.assertEqual(len(metadata['records']),1)
        self.assertEqual(arrays['cohort_context'].shape,(2,12,6))
        self.assertTrue(metadata['cohort']['enabled'])
        self.assertEqual(list(arrays['record_index']),[0,0]);self.assertEqual(list(arrays['sample_index']),[0,1])
        self.assertEqual(len(metadata['rows']),2)

    def test_duplicate_individual_bam_or_sm_is_rejected(self):
        original=self.bam(name='S');copy=self.root/'copy.bam'
        shutil.copyfile(original,copy);shutil.copyfile(str(original)+'.bai',str(copy)+'.bai')
        for label,path,message in [('samepath',original,'Same BAM'),('sameSM',copy,'SM identifiers')]:
            p=argparse.ArgumentParser();add_inputs(p);p.add_argument('--out');p.add_argument('--no-cohort',action='store_true')
            args=p.parse_args(['--bam','S='+str(original),'--bam','T='+str(path),'--reference',str(self.ref),
                               '--region','chr1:101-300','--panel-reader','python','--out',str(self.root/label)])
            with self.assertRaisesRegex(ValueError,message):prepare(args)


if __name__=='__main__':unittest.main()
