import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import pysam
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'base'))
from joint_ld import project_donors,collect_components
from local_alleles import make_blocks

class JointIntegrationContracts(unittest.TestCase):
    def test_linked_snp_indel_donors_and_unrepresented_path(self):
        sequence='ACGTACGT';fetch=lambda s,e:sequence[s:e]
        seeds=[(2,'G','T'),(5,'C','CA')]
        row=dict(start0=2,end0=6,evidence_start0=2,evidence_end0=6,
            alleles=['GTAC','TTAC','GTACA'],seed_variants=[dict(start0=p,ref=r,alt=a) for p,r,a in seeds])
        panel={seeds[0]:(np.array([0,1,0,1]),.5),seeds[1]:(np.array([0,0,1,1]),.5)}
        actual=project_donors([(None,row)],panel,np.arange(4),fetch)
        np.testing.assert_array_equal(actual,[[0,1,2,255]])

    def test_component_ownership_and_sample_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);features=p/'features';features.mkdir()
            records=[dict(record_id=f'chr1:{pos+1}:A:C',start0=pos,alleles=['A','C']) for pos in [10,20]]
            (features/'metadata.json').write_text(json.dumps(dict(records=records,base=dict(excluded_panel_samples=['original_SM','relative']))))
            h=pysam.VariantHeader();h.contigs.add('chr1',length=100);h.add_sample('alias')
            vcf=p/'calls.vcf.gz'
            with pysam.VariantFile(vcf,'wz',header=h) as out:
                for pos in [10,20]:out.write(out.new_record(contig='chr1',start=pos,alleles=['A','C']))
            component=dict(neural=str(vcf),features=[str(features)],start0=15,end0=25)
            entries,header,inputs,excluded=collect_components([component])
            self.assertEqual(len(entries),1);self.assertEqual(entries[0][0].start,20)
            self.assertIn('original_SM',excluded);self.assertIn('relative',excluded)
            self.assertEqual(list(header.samples),['alias'])
            with self.assertRaisesRegex(ValueError,'ownership'):collect_components([component,component])

    def test_read_pattern_ties_do_not_depend_on_fragment_iteration_order(self):
        variants=[(p,'A','C') for p in range(10,14)]
        panel={v:(np.zeros(8),0.) for v in variants}
        fragments={variants[0]:['a','b','c','d'],variants[1]:['a','b'],variants[2]:['c','d']}
        reverse={key:list(reversed(value)) for key,value in fragments.items()}
        kwargs=dict(fetch=lambda a,b:'A'*(b-a),panel=panel,support={v:4 for v in variants},max_alleles=4,groups_override=[[0,1,2,3]])
        a=make_blocks(variants,fragment_seeds=fragments,**kwargs)
        b=make_blocks(variants,fragment_seeds=reverse,**kwargs)
        self.assertEqual(a[0]['alleles'],b[0]['alleles'])

if __name__=='__main__':unittest.main()
