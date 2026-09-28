import sys,tempfile,unittest
from pathlib import Path
import pysam
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'caller'))
from snv_event_ownership import reconcile
from ownership_events import catalog_events

class SNVEventProjection(unittest.TestCase):
    def test_duplicate_snp_removed_but_unique_indel_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);ref='ACGTGCAATGCTACGATCGTACCGTAGCATGACTGATCGATCGTACGTAGCTAGCTACGTA'
            fa=d/'ref.fa';fa.write_text('>chr1\n'+ref+'\n');pysam.faidx(str(fa))
            header=pysam.VariantHeader();header.contigs.add('chr1',length=len(ref));header.add_sample('S')
            for name,number,kind in [('GT',1,'String'),('GP','G','Float'),('GQ',1,'Integer'),('DP',1,'Integer'),('CS',1,'Float'),('DS','A','Float')]:header.formats.add(name,number,kind,name)
            header.filters.add('LowQual',None,None,'low');vc=d/'in.vcf.gz'
            alt=ref[10:15]+ref[16:30];base=next(a for a in 'ACGT' if a!=ref[25]);alt=alt[:14]+base+alt[15:]
            with pysam.VariantFile(vc,'wz',header=header) as out:
                for start,alleles,gp in [(10,[ref[10:30],alt],(.1,.8,.1)),(25,[ref[25],base],(.0005,.999,.0005))]:
                    v=header.new_record(contig='chr1',start=start,alleles=alleles);v.samples['S']['GT']=(0,1);v.samples['S']['GP']=gp;v.samples['S']['DP']=40;v.filter.add('PASS');out.write(v)
            pysam.tabix_index(str(vc),preset='vcf',force=True);dest=d/'out.vcf.gz';receipt=reconcile(vc,fa,dest)
            self.assertEqual(receipt['counts']['removed_duplicate_events'],1)
            called=[]
            with pysam.FastaFile(fa) as reference,pysam.VariantFile(dest) as source:
                for v in source:
                    catalog=catalog_events(v.start,list(v.alleles),lambda s,e:reference.fetch('chr1',s,e).upper())
                    called.extend(event for allele in set(v.samples['S']['GT']) for event in catalog[allele])
                    self.assertAlmostEqual(sum(v.samples['S']['GP']),1,places=6)
            self.assertEqual(sum(len(r)!=len(a) for p,r,a in called),1)
            self.assertEqual(sum(len(r)==len(a)==1 for p,r,a in called),1)
if __name__=='__main__':unittest.main()
