import sys,unittest
from pathlib import Path
import pysam
sys.path.insert(0,str(Path(__file__).parent/'caller/caller'))
from phase_query import relationship

class QueryTests(unittest.TestCase):
    def records(self):
        h=pysam.VariantHeader();h.contigs.add('chr1');h.add_sample('S')
        for n,t in [('GT','String'),('PS','Integer'),('VPE','Float')]:h.formats.add(n,1,t,n)
        a=h.new_record(contig='chr1',start=10,alleles=('A','G'));b=h.new_record(contig='chr1',start=1000,alleles=('AC','A'))
        for r in [a,b]:r.samples['S']['GT']=(0,1);r.samples['S'].phased=True;r.samples['S']['PS']=11;r.samples['S']['VPE']=.002
        return a,b
    def test_cis_and_trans(self):
        a,b=self.records();self.assertEqual(relationship(a,1,b,1,'S')['result'],'cis')
        b.samples['S']['GT']=(1,0);b.samples['S'].phased=True
        self.assertEqual(relationship(a,1,b,1,'S')['result'],'trans')
    def test_block_boundary_unknown(self):
        a,b=self.records();b.samples['S']['PS']=1001
        self.assertEqual(relationship(a,1,b,1,'S')['result'],'unknown')
    def test_homozygote_unknown(self):
        a,b=self.records();b.samples['S']['GT']=(1,1)
        self.assertEqual(relationship(a,1,b,1,'S')['result'],'unknown')
    def test_unphased_unknown(self):
        a,b=self.records();b.samples['S'].phased=False
        self.assertEqual(relationship(a,1,b,1,'S')['result'],'unknown')

if __name__=='__main__':unittest.main()
