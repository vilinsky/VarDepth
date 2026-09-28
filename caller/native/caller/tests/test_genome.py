"""Scheduler mechanics only: synthetic VCFs are not genomic accuracy evidence."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import pysam

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import genome


def make_vcf(path, chrom='chr1', rows=(), samples=('a', 'b'), length=1000, gp_number='G'):
    header = pysam.VariantHeader()
    header.contigs.add(chrom, length=length)
    header.add_meta('FORMAT', items=[('ID', 'GT'), ('Number', 1), ('Type', 'String'), ('Description', 'Genotype')])
    header.add_meta('FORMAT', items=[('ID', 'GP'), ('Number', gp_number), ('Type', 'Float'), ('Description', 'Genotype probability')])
    for sample in samples: header.add_sample(sample)
    path = Path(path)
    with pysam.VariantFile(str(path), 'wz', header=header) as out:
        for start, ref, alt in rows:
            rec = out.new_record(contig=chrom, start=start, alleles=(ref, alt))
            for index, sample in enumerate(samples):
                rec.samples[sample]['GT'] = (0, 1) if index == 0 else (1, 1)
                rec.samples[sample]['GP'] = (.1, .8, .1) if index == 0 else (.1, .1, .8)
            out.write(rec)
    pysam.tabix_index(str(path), preset='vcf', force=True)
    return path


class GenomeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.lengths = {'chr1': 1000, 'chr2': 700}
        self.samples = ['a', 'b']

    def tearDown(self): self.tmp.cleanup()

    def test_plan_merges_union_and_clips_padding_in_reference_order(self):
        chunks = genome.plan_chunks(['chr2:51-100', 'chr1:901-1000', 'chr1:1-150',
                                     'chr1:101-200', 'chr1:201-250'], self.lengths, 100, 200)
        self.assertEqual([(c.chrom, c.core_start0, c.core_end0) for c in chunks],
                         [('chr1', 0, 100), ('chr1', 100, 200), ('chr1', 200, 250),
                          ('chr1', 900, 1000), ('chr2', 50, 100)])
        self.assertEqual((chunks[0].padded_start0, chunks[0].padded_end0), (0, 300))
        self.assertEqual(chunks[3].padded_end0, 1000)
        self.assertEqual(chunks[-1].region, 'chr2:1-300')
        self.assertEqual(chunks, genome.plan_chunks(['chr1:1-250', 'chr2:51-100', 'chr1:901-1000'], self.lengths, 100, 200))

    def test_invalid_regions_and_chunk_settings(self):
        for region in ['chr1:0-50', 'chr1:1-1001', 'chr1:100-1', 'chr3:1-20', 'chr1:1']:
            with self.subTest(region=region), self.assertRaises(ValueError): genome.parse_region(region, self.lengths)
        with self.assertRaises(ValueError): genome.plan_chunks(['chr1:1-10'], self.lengths, 0)
        with self.assertRaises(ValueError): genome.plan_chunks(['chr1:1-10'], self.lengths, 10, -1)

    def test_owned_normalized_start_seam_and_whole_reference_allele(self):
        chunks = genome.plan_chunks(['chr1:1-200'], self.lengths, 100, 100)
        # Normalization has already placed the deletion at 98; its raw seed
        # could have been on the right. A REF spanning the seam stays intact.
        rows = [(98, 'AAAAA', 'A'), (99, 'C', 'T'), (100, 'G', 'A'), (150, 'A', 'C')]
        paths = []
        for index, chunk in enumerate(chunks):
            padded = make_vcf(self.root / f'p{index}.vcf.gz', rows=rows)
            owned = self.root / f'c{index}.vcf.gz'
            counts = genome.filter_owned_core(padded, owned, chunk, self.samples, self.lengths)
            self.assertEqual(counts, {'owned_records': 2, 'outside_core_records': 2})
            paths.append(owned)
        output = self.root / 'merged.vcf.gz'
        self.assertEqual(genome.merge_owned_vcfs(chunks, paths, output, self.samples, self.lengths)['records'], 4)
        with pysam.VariantFile(str(output)) as variants:
            records = list(variants.fetch('chr1'))
            self.assertEqual([r.start for r in records], [98, 99, 100, 150])
            self.assertEqual(records[0].ref, 'AAAAA')
            self.assertEqual(records[0].stop, 103)

    def test_union_contigs_translation_preserves_genotypes_and_probabilities(self):
        chunks = genome.plan_chunks(['chr2:1-200', 'chr1:1-200'], self.lengths, 200, 100)
        paths = [make_vcf(self.root / f'{c.chrom}.vcf.gz', chrom=c.chrom,
                          rows=[(10, 'A', 'C')], length=self.lengths[c.chrom]) for c in chunks]
        output = self.root / 'union.vcf.gz'
        genome.merge_owned_vcfs(chunks, paths, output, self.samples, self.lengths)
        with pysam.VariantFile(str(output)) as variants:
            self.assertEqual(list(variants.header.contigs), ['chr1', 'chr2'])
            self.assertEqual(list(variants.header.samples), self.samples)
            rows = list(variants)
            self.assertEqual([r.contig for r in rows], ['chr1', 'chr2'])
            for row in rows:
                self.assertEqual(row.samples['a']['GT'], (0, 1))
                self.assertEqual(row.samples['b']['GT'], (1, 1))
                self.assertAlmostEqual(row.samples['b']['GP'][2], .8)
        with pysam.VariantFile(str(output)) as variants:
            self.assertEqual(len(list(variants.fetch('chr2', 0, 30))), 1)

    def test_sample_order_missing_sample_and_schema_conflicts_rejected(self):
        chunks = genome.plan_chunks(['chr1:1-200'], self.lengths, 100, 100)
        good = make_vcf(self.root / 'good.vcf.gz', rows=[(10, 'A', 'C')])
        for label, samples, number in [('order', ['b', 'a'], 'G'), ('missing', ['a'], 'G'), ('format', ['a', 'b'], '.')]:
            bad = make_vcf(self.root / f'{label}.vcf.gz', rows=[(110, 'A', 'C')], samples=samples, gp_number=number)
            output = self.root / 'unchanged.vcf.gz'; output.write_bytes(b'previous-valid-output')
            with self.subTest(label=label), self.assertRaises(ValueError):
                genome.merge_owned_vcfs(chunks, [good, bad], output, self.samples, self.lengths)
            self.assertEqual(output.read_bytes(), b'previous-valid-output')

    def test_duplicate_rejection_preserves_previous_publication(self):
        chunk = genome.plan_chunks(['chr1:1-200'], self.lengths, 200)[0]
        path = make_vcf(self.root / 'duplicate.vcf.gz', rows=[(10, 'A', 'C'), (10, 'A', 'C')])
        output = self.root / 'out.vcf.gz'; output.write_bytes(b'previous')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            genome.merge_owned_vcfs([chunk], [path], output, self.samples, self.lengths)
        self.assertEqual(output.read_bytes(), b'previous')
        self.assertFalse(any('.partial' in p.name for p in self.root.iterdir()))

    def test_overlapping_owned_cores_rejected(self):
        chunk = genome.plan_chunks(['chr1:1-200'], self.lengths, 200)[0]
        path = make_vcf(self.root / 'one.vcf.gz', rows=[(10, 'A', 'C')])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            genome.merge_owned_vcfs([chunk, chunk], [path, path], self.root/'out.vcf', self.samples, self.lengths)

    def fake_runner(self, calls, fail=False, wrong_sha=False):
        def run(command, **kwargs):
            calls.append(command)
            self.assertTrue(kwargs['check'])
            output = Path(command[command.index('--output') + 1])
            make_vcf(output, rows=[(10, 'A', 'C'), (150, 'C', 'T')])
            if fail: raise subprocess.CalledProcessError(42, command)
            (Path(str(output)+'.pipeline.json')).write_text(json.dumps({'output_sha256': 'bad' if wrong_sha else genome.sha(output)}))
            (Path(str(output)+'.manifest.json')).write_text(json.dumps({'synthetic': True}))
            (Path(str(output)+'.posterior.npz')).write_bytes(b'synthetic-test-sidecar')
        return run

    def test_completed_chunks_reused_only_after_all_artifact_hashes(self):
        chunk = genome.plan_chunks(['chr1:1-100'], self.lengths, 100)[0]
        directory = self.root/'chunk'; calls = []
        kwargs = dict(chunk=chunk, directory=directory, run_key='key', base_command=['mock-pipeline'],
                      sample_ids=self.samples, lengths=self.lengths, runner=self.fake_runner(calls))
        self.assertFalse(genome.run_chunk(**kwargs)['reused'])
        self.assertTrue(genome.run_chunk(**kwargs)['reused'])
        self.assertEqual(len(calls), 1)
        (directory/'padded.vcf.gz.posterior.npz').write_bytes(b'tampered')
        self.assertIsNone(genome.verified_completion(directory, 'key', chunk, self.samples, self.lengths))
        self.assertFalse(genome.run_chunk(**kwargs)['reused'])
        self.assertEqual(len(calls), 2)
        self.assertIsNone(genome.verified_completion(directory, 'other-key', chunk, self.samples, self.lengths))

    def test_failure_resume_removes_stale_completion_and_preserves_evidence(self):
        chunk = genome.plan_chunks(['chr1:1-100'], self.lengths, 100)[0]
        directory = self.root/'chunk'; calls = []
        kwargs = dict(chunk=chunk, directory=directory, run_key='key', base_command=['mock'],
                      sample_ids=self.samples, lengths=self.lengths)
        genome.run_chunk(**kwargs, runner=self.fake_runner(calls))
        (directory/'padded.vcf.gz.manifest.json').write_text('bad')
        with self.assertRaises(subprocess.CalledProcessError):
            genome.run_chunk(**kwargs, runner=self.fake_runner(calls, fail=True))
        self.assertFalse((directory/'COMPLETE.json').exists())
        self.assertTrue((directory/'FAILED.json').exists())
        self.assertFalse(genome.run_chunk(**kwargs, runner=self.fake_runner(calls))['reused'])
        self.assertFalse((directory/'FAILED.json').exists())
        self.assertTrue(genome.run_chunk(**kwargs, runner=self.fake_runner(calls))['reused'])
        self.assertEqual(len(calls), 3)

    def test_pipeline_claimed_sha_must_match_actual_vcf(self):
        chunk = genome.plan_chunks(['chr1:1-100'], self.lengths, 100)[0]
        directory = self.root/'chunk'
        with self.assertRaisesRegex(ValueError, 'checksum'):
            genome.run_chunk(chunk, directory, 'key', ['mock'], self.samples, self.lengths,
                             runner=self.fake_runner([], wrong_sha=True))
        self.assertFalse((directory/'COMPLETE.json').exists())

    def test_scheduler_limits_submission_and_drains_after_failure(self):
        chunks = genome.plan_chunks(['chr1:1-500'], self.lengths, 100)
        both_started = threading.Barrier(2)
        first_failed = threading.Event(); second_finished = threading.Event()
        observed = []; lock = threading.Lock()
        def worker(chunk):
            with lock: observed.append(chunk.id)
            if chunk not in chunks[:2]: self.fail('Submitted a new job after failure')
            both_started.wait(timeout=5)
            if chunk == chunks[0]:
                first_failed.set()
                raise RuntimeError('intentional chunk failure')
            self.assertTrue(first_failed.wait(5))
            second_finished.set()
            return {'complete': True}
        with self.assertRaisesRegex(RuntimeError, 'drained'):
            genome.schedule_chunks(chunks, worker, jobs=2)
        self.assertTrue(second_finished.is_set())
        self.assertCountEqual(observed, [chunks[0].id, chunks[1].id])

    def test_scheduler_returns_plan_order_and_bounds_live_workers(self):
        chunks = genome.plan_chunks(['chr1:1-600'], self.lengths, 100)
        lock = threading.Lock(); active = 0; peak = 0
        def worker(chunk):
            nonlocal active, peak
            with lock: active += 1; peak = max(active, peak)
            with lock: active -= 1
            return chunk.id
        self.assertEqual(genome.schedule_chunks(chunks, worker, 2), [c.id for c in chunks])
        self.assertLessEqual(peak, 2)

    def test_main_bam_to_vcf_plumbing_resume_and_atomic_failure(self):
        # Fake only the expensive pipeline process. Argument propagation,
        # hashing, scheduling, filtering, merging and resume are real.
        reference = self.root/'ref.fa'; reference.write_text('>chr1\n' + 'A'*1000 + '\n')
        Path(str(reference)+'.fai').write_text('chr1\t1000\t6\t1000\t1001\n')
        checkpoint = self.root/'model.pt'; checkpoint.write_bytes(b'synthetic checkpoint')
        bams = [self.root/'a.bam', self.root/'b.bam']
        for bam in bams: bam.write_bytes(b'synthetic BAM; process boundary mocked')
        output = self.root/'final.vcf.gz'; work = self.root/'work'
        argv = ['genome.py', '--bam', 'a='+str(bams[0]), '--bam', 'b='+str(bams[1]),
                '--reference', str(reference), '--checkpoint', str(checkpoint),
                '--regions', 'chr1:1-200', '--output', str(output), '--work-dir', str(work),
                '--chunk-bp', '100', '--padding-bp', '150', '--no-ld-proposals']
        original = genome.run_chunk; calls = []
        def execute(*args, **kwargs): return original(*args, **kwargs, runner=self.fake_runner(calls))
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(genome, 'run_chunk', side_effect=execute):
            genome.main()
            old_sha = genome.sha(output)
            genome.main()
        self.assertEqual(len(calls), 2)
        self.assertTrue(all('--no-ld-proposals' in command for command in calls))
        manifest = json.loads(Path(str(output)+'.genome.json').read_text())
        self.assertEqual(manifest['reused_chunks'], 2)
        self.assertEqual(manifest['records'], 2)
        self.assertEqual(manifest['output_sha256'], old_sha)
        self.assertEqual(manifest['identity']['samples'], self.samples)
        receipt = next(work.glob('run-*/*/COMPLETE.json'))
        (receipt.parent/'padded.vcf.gz.manifest.json').write_text('tampered')
        def fail(*args, **kwargs): return original(*args, **kwargs, runner=self.fake_runner(calls, fail=True))
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(genome, 'run_chunk', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'failed'): genome.main()
        self.assertEqual(genome.sha(output), old_sha)
        self.assertEqual(json.loads(Path(str(output)+'.genome.json').read_text()), manifest)

    def test_plan_only_cli_defaults_and_minimum_padding(self):
        reference = self.root/'ref.fa'
        Path(str(reference)+'.fai').write_text('chr1\t1000\t6\t1000\t1001\n')
        argv = ['genome.py', '--bam', 'a=not-read.bam', '--reference', str(reference),
                '--checkpoint', 'not-read.pt', '--output', str(self.root/'out.vcf'),
                '--work-dir', str(self.root/'work'), '--contigs', 'chr1', '--plan-only']
        with mock.patch.object(sys, 'argv', argv):
            self.assertEqual(genome.build_parser().parse_args().padding_bp, 200000)
            genome.main()
        with mock.patch.object(sys, 'argv', argv + ['--padding-bp', '149']):
            with self.assertRaisesRegex(ValueError, 'max-indel'): genome.main()


if __name__ == '__main__': unittest.main()
