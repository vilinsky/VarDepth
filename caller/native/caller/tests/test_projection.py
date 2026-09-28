import unittest
import random
import numpy as np
import pysam
from projection import project,common_anchors,crop_observation,expand_compound_context


def read(seq,start,cigar):
    r=pysam.AlignedSegment();r.query_name='molecule';r.query_sequence=seq
    r.reference_id=0;r.reference_start=start;r.mapping_quality=60;r.cigarstring=cigar
    r.query_qualities=[35]*len(seq);return r


class ProjectionContracts(unittest.TestCase):
    def test_compound_snp_and_indel_extends_the_repeat_context(self):
        full='CGTGCT'+20*'A'+'GCTAGCTA';start=1002;reference=full[2:12]
        alternate=reference[:1]+'A'+reference[2:6]+reference[7:]
        fetch=lambda s,e:full[max(0,s-1000):min(len(full),e-1000)]
        new_start,alleles=expand_compound_context(start,[reference,alternate],fetch)
        self.assertGreater(new_start+len(alleles[0]),1026)
        self.assertEqual(len(alleles[0])-len(alleles[1]),1)
        self.assertEqual(alleles[0],fetch(new_start,new_start+len(alleles[0])))

    def test_short_full_context_retains_spanning_indel_evidence(self):
        full='CGTACGATCAGTC';start=1002;reference=full[2:11]
        alternate=reference[:4]+'T'+reference[4:]
        fetch=lambda s,e:full[s-1000:e-1000]
        maps,safe=common_anchors(start,reference,[reference,alternate],fetch)
        r=read(alternate,start,'4M1I5M')
        obs=crop_observation(r,start,reference,[reference,alternate],maps,safe)
        self.assertIsNotNone(obs)
        self.assertEqual(obs['sequence'],obs['haplotypes'][1])

    def test_partial_read_retains_true_insertion(self):
        rng=random.Random(19);reference=''.join(rng.choices('ACGT',k=220))
        alternate=reference[:80]+'TT'+reference[80:];start=1000
        fetch=lambda s,e:reference[max(0,s-start):min(len(reference),e-start)]
        maps,safe=common_anchors(start,reference,[reference,alternate],fetch)
        r=read(reference[50:80]+'TT'+reference[80:160],start+50,'30M2I80M')
        obs=crop_observation(r,start,reference,[reference,alternate],maps,safe)
        self.assertIsNotNone(obs);self.assertFalse(obs['whole_context'])
        self.assertEqual(obs['haplotypes'][1],obs['sequence'])
        self.assertNotEqual(obs['haplotypes'][0],obs['sequence'])

    def test_unspanned_repeat_does_not_become_reference_evidence(self):
        reference='CGTCGTCGTC'+30*'A'+'CGTCGTCGTC';alternate=reference[:20]+'A'+reference[20:]
        start=1000;fetch=lambda s,e:reference[max(0,s-start):min(len(reference),e-start)]
        maps,safe=common_anchors(start,reference,[reference,alternate],fetch)
        r=read(reference[15:50],start+15,'35M')
        self.assertIsNone(crop_observation(r,start,reference,[reference,alternate],maps,safe))

    def test_hidden_variant_has_identical_partial_sequence(self):
        rng=random.Random(61);reference=''.join(rng.choices('ACGT',k=220));start=1000
        first=reference[:80]+'TT'+reference[80:]
        second=first[:190]+('A' if first[190]!='A' else 'C')+first[191:]
        fetch=lambda s,e:reference[max(0,s-start):min(len(reference),e-start)]
        alleles=[reference,first,second];maps,safe=common_anchors(start,reference,alleles,fetch)
        r=read(reference[50:80]+'TT'+reference[80:160],start+50,'30M2I80M')
        obs=crop_observation(r,start,reference,alleles,maps,safe)
        self.assertEqual(obs['haplotypes'][1],obs['haplotypes'][2])

    def test_deletion_maps_no_base_to_deleted_reference(self):
        mapping,events=project('ACGTGCA','ACTGCA')
        self.assertEqual(int((mapping==-1).sum()),1)
        self.assertEqual(sum(len(r) for p,r,a in events),1)
        kept=mapping[mapping>=0];np.testing.assert_array_equal(kept,np.arange(6))


if __name__=='__main__':unittest.main()
