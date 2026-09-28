from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline import feature_fingerprint


class CacheTests(unittest.TestCase):
    def test_stable_json_key_order(self):
        self.assertEqual(feature_fingerprint({'b':2,'a':1},{'x':{'sha':'1'}},{'core':'2'})[0],
                         feature_fingerprint({'a':1,'b':2},{'x':{'sha':'1'}},{'core':'2'})[0])

    def test_region_input_source_and_cohort_change_cache(self):
        base={'region':'chr1:1-10','no_cohort':False}
        key=feature_fingerprint(base,{'bam':'a'},{'core':'1'})[0]
        for args,inputs,source in [({**base,'region':'chr1:11-20'},{'bam':'a'},{'core':'1'}),
                                   (base,{'bam':'b'},{'core':'1'}),
                                   (base,{'bam':'a'},{'core':'2'}),
                                   ({**base,'no_cohort':True},{'bam':'a'},{'core':'1'})]:
            self.assertNotEqual(key,feature_fingerprint(args,inputs,source)[0])


if __name__=='__main__':unittest.main()
