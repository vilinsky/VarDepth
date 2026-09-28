#!/usr/bin/env python3
"""Restartable padded-region scheduler for pipeline.py (Illumina/cohort defaults).

Chunked LD/state selection and cohort context are approximations: finite padding
is not equivalent to chromosome-wide inference. Ownership uses the final,
normalized VCF start, not read/seed position; a record spanning a seam remains
whole and belongs to one core. Extreme normalization shifts or different local
candidate graphs can still affect completeness at seams.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, dataclass
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
VERSION = 'genome-scheduler-0.1.0'
ARTIFACTS = ('padded.vcf.gz', 'padded.vcf.gz.tbi', 'padded.vcf.gz.pipeline.json',
             'padded.vcf.gz.manifest.json', 'padded.vcf.gz.posterior.npz',
             'core.vcf.gz', 'core.vcf.gz.tbi')


@dataclass(frozen=True)
class Chunk:
    id: str
    chrom: str
    core_start0: int
    core_end0: int
    padded_start0: int
    padded_end0: int

    @property
    def region(self):
        return f'{self.chrom}:{self.padded_start0 + 1}-{self.padded_end0}'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        with temporary.open('w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def reference_lengths(reference):
    """Read an existing FASTA index; do not silently scan a whole reference."""
    index = Path(str(reference) + '.fai')
    if not index.exists(): raise ValueError('Indexed FASTA required (.fai missing): ' + str(reference))
    lengths = {}
    for line in index.read_text().splitlines():
        fields = line.split('\t')
        if len(fields) < 2 or fields[0] in lengths: raise ValueError('Invalid/duplicate FASTA index contig')
        length = int(fields[1])
        if length < 1: raise ValueError('FASTA contig must have positive length')
        lengths[fields[0]] = length
    if not lengths: raise ValueError('Empty FASTA index')
    return lengths


def parse_region(text, lengths):
    try:
        chrom, interval = text.rsplit(':', 1)
        start, end = map(int, interval.split('-'))
    except (ValueError, AttributeError):
        raise ValueError('Region must be CHROM:START-END, 1-based inclusive') from None
    if chrom not in lengths or start < 1 or end < start or end > lengths[chrom]:
        raise ValueError('Region outside indexed reference: ' + str(text))
    return chrom, start - 1, end


def plan_chunks(regions, lengths, chunk_bp=1_000_000, padding_bp=200_000):
    """Merge requested overlaps, then tile disjoint half-open owned cores."""
    if chunk_bp < 1 or padding_bp < 0: raise ValueError('Invalid chunk size/padding')
    rank = {chrom: i for i, chrom in enumerate(lengths)}
    intervals = []
    for text in regions:
        intervals.append(parse_region(text, lengths) if isinstance(text, str) else tuple(text))
    for chrom, start, end in intervals:
        if chrom not in rank or not 0 <= start < end <= lengths[chrom]: raise ValueError('Invalid owned interval')
    merged = []
    for chrom, start, end in sorted(intervals, key=lambda r: (rank[r[0]], r[1], r[2])):
        if merged and chrom == merged[-1][0] and start <= merged[-1][2]:
            merged[-1] = (chrom, merged[-1][1], max(end, merged[-1][2]))
        else: merged.append((chrom, start, end))
    result = []
    for chrom, start, end in merged:
        for core_start in range(start, end, chunk_bp):
            core_end = min(end, core_start + chunk_bp)
            identity = [chrom, core_start, core_end]
            key = f'{len(result):06d}-' + fingerprint(identity)[:12]
            result.append(Chunk(key, chrom, core_start, core_end,
                                max(0, core_start - padding_bp), min(lengths[chrom], core_end + padding_bp)))
    return result


def _header_signature(header):
    signature = {}
    for name in ('info', 'formats', 'filters'):
        values = getattr(header, name)
        signature[name] = {key: (str(getattr(values[key], 'number', '')), str(getattr(values[key], 'type', '')),
                                 str(values[key].description)) for key in values}
    return signature


def _check_header(header, sample_ids, lengths, signature=None):
    if header.version != 'VCFv4.2':
        raise ValueError('Scheduler expects the pipeline VCFv4.2 schema')
    if list(header.samples) != list(sample_ids):
        raise ValueError('Chunk sample header/order differs from the requested people')
    for chrom in header.contigs:
        length = header.contigs[chrom].length
        if chrom not in lengths or (length is not None and length != lengths[chrom]):
            raise ValueError('Chunk contig header differs from reference: ' + chrom)
    if signature is not None and _header_signature(header) != signature:
        raise ValueError('Chunk INFO/FORMAT/FILTER schemas disagree')


def filter_owned_core(source, target, chunk, sample_ids, lengths):
    import pysam
    target = Path(target)
    temporary = target.with_name('.' + target.stem + '.' + uuid.uuid4().hex + '.vcf.gz')
    kept = dropped = 0
    previous = -1
    try:
        with pysam.VariantFile(str(source)) as inp:
            _check_header(inp.header, sample_ids, lengths)
            with pysam.VariantFile(str(temporary), 'wz', header=inp.header) as out:
                for record in inp:
                    if record.contig != chunk.chrom: raise ValueError('Unexpected contig in chunk records')
                    if record.start < previous: raise ValueError('Padded VCF is not coordinate sorted')
                    previous = record.start
                    if chunk.core_start0 <= record.start < chunk.core_end0:
                        out.write(record); kept += 1
                    else: dropped += 1
        pysam.tabix_index(str(temporary), preset='vcf', force=True)
        os.replace(temporary, target)
        os.replace(str(temporary) + '.tbi', str(target) + '.tbi')
    finally:
        temporary.unlink(missing_ok=True)
        Path(str(temporary) + '.tbi').unlink(missing_ok=True)
    return {'owned_records': kept, 'outside_core_records': dropped}


def verified_completion(directory, run_key, chunk, sample_ids, lengths):
    """A marker alone never qualifies: verify identity, all artifact hashes and VCF."""
    import pysam
    directory = Path(directory)
    try:
        receipt = json.loads((directory / 'COMPLETE.json').read_text())
        if receipt['run_key'] != run_key or receipt['chunk'] != asdict(chunk): return None
        if set(receipt['artifacts']) != set(ARTIFACTS): return None
        for name in ARTIFACTS:
            path = directory / name
            expected = receipt['artifacts'][name]
            if not path.is_file() or path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
                return None
        report = json.loads((directory / 'padded.vcf.gz.pipeline.json').read_text())
        if report['output_sha256'] != receipt['artifacts']['padded.vcf.gz']['sha256']: return None
        with pysam.VariantFile(str(directory / 'core.vcf.gz')) as variants:
            _check_header(variants.header, sample_ids, lengths)
            previous = -1
            for record in variants:
                if record.contig != chunk.chrom or not chunk.core_start0 <= record.start < chunk.core_end0:
                    return None
                if record.start < previous: return None
                previous = record.start
        return receipt
    except (OSError, ValueError, KeyError, TypeError):
        return None


def run_chunk(chunk, directory, run_key, base_command, sample_ids, lengths, *, runner=subprocess.run):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'chunk.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        old = verified_completion(directory, run_key, chunk, sample_ids, lengths)
        if old is not None: return {**old, 'reused': True}
        # Stale completion must not be mistaken for success if rerunning fails.
        (directory / 'COMPLETE.json').unlink(missing_ok=True)
        command = [*base_command, '--region', chunk.region, '--output', str(directory / 'padded.vcf.gz'),
                   '--work-dir', str(directory / 'pipeline-work')]
        if any(Path(str(x)).name == 'run_v05.py' for x in base_command):
            command += ['--core-region', f'{chunk.chrom}:{chunk.core_start0 + 1}-{chunk.core_end0}']
        started = time.monotonic()
        try:
            with (directory / 'pipeline.log').open('w') as log:
                runner(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            counts = filter_owned_core(directory / 'padded.vcf.gz', directory / 'core.vcf.gz',
                                       chunk, sample_ids, lengths)
            artifacts = {name: {'bytes': (directory / name).stat().st_size, 'sha256': sha(directory / name)}
                         for name in ARTIFACTS}
            pipeline_report = json.loads((directory / 'padded.vcf.gz.pipeline.json').read_text())
            if pipeline_report['output_sha256'] != artifacts['padded.vcf.gz']['sha256']:
                raise ValueError('Pipeline manifest/output checksum disagreement')
            receipt = dict(run_key=run_key, chunk=asdict(chunk), artifacts=artifacts, command=command,
                           counts=counts, seconds=time.monotonic() - started,
                           completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            atomic_json(directory / 'COMPLETE.json', receipt)
            (directory / 'FAILED.json').unlink(missing_ok=True)
            return {**receipt, 'reused': False}
        except Exception as error:
            atomic_json(directory / 'FAILED.json', dict(chunk=asdict(chunk), run_key=run_key,
                                                       error=str(error), command=command))
            raise


def schedule_chunks(chunks, worker, jobs=1):
    """Bound submissions to jobs. On failure stop submitting and drain current work."""
    if jobs < 1: raise ValueError('jobs must be positive')
    pending_chunks = iter(chunks); results = {}; failures = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        running = {}
        for _ in range(min(jobs, len(chunks))):
            chunk = next(pending_chunks); running[executor.submit(worker, chunk)] = chunk
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda f: running[f].id):
                chunk = running.pop(future)
                try: results[chunk.id] = future.result()
                except Exception as error: failures.append((chunk.id, str(error)))
            if not failures:
                for _ in range(jobs - len(running)):
                    chunk = next(pending_chunks, None)
                    if chunk is None: break
                    running[executor.submit(worker, chunk)] = chunk
    if failures:
        raise RuntimeError('Chunk execution failed; active jobs drained, no new jobs submitted: ' + repr(failures))
    return [results[chunk.id] for chunk in chunks]


def merge_owned_vcfs(chunks, paths, output, sample_ids, lengths):
    """Validate all headers before publication; reject duplicate exact variant keys."""
    import pysam
    if not chunks or len(chunks) != len(paths): raise ValueError('Chunks and core VCFs must align')
    rank = {chrom: i for i, chrom in enumerate(lengths)}
    signature = first_header = None
    previous_core = None
    for chunk, path in zip(chunks, paths):
        if previous_core and (rank[chunk.chrom], chunk.core_start0) < (rank[previous_core.chrom], previous_core.core_end0):
            raise ValueError('Owned cores overlap or are out of reference order')
        previous_core = chunk
        with pysam.VariantFile(str(path)) as variants:
            _check_header(variants.header, sample_ids, lengths, signature)
            if first_header is None:
                first_header = variants.header.copy(); signature = _header_signature(variants.header)
    header = pysam.VariantHeader()
    for record in first_header.records:
        line = str(record).strip()
        if line.startswith(('##fileformat=', '##contig=', '##FILTER=<ID=PASS,')): continue
        header.add_line(line)
    included = {chunk.chrom for chunk in chunks}
    for chrom, length in lengths.items():
        if chrom in included: header.contigs.add(chrom, length=length)
    for sample in sample_ids: header.add_sample(sample)
    header.add_meta('chunked_inference', value='Approximate padded LD/context; normalized start0 owns each record; see genome manifest')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    compressed = str(output).endswith('.vcf.gz')
    if not compressed and not str(output).endswith('.vcf'): raise ValueError('Output must end in .vcf or .vcf.gz')
    temporary = output.with_name('.' + output.name + '.' + uuid.uuid4().hex + ('.vcf.gz' if compressed else '.vcf'))
    count = 0; previous_position = None; keys_at_position = set()
    try:
        with pysam.VariantFile(str(temporary), 'wz' if compressed else 'w', header=header) as out:
            for chunk, path in zip(chunks, paths):
                with pysam.VariantFile(str(path)) as variants:
                    for record in variants:
                        if record.contig != chunk.chrom or not chunk.core_start0 <= record.start < chunk.core_end0:
                            raise ValueError('Record lies outside its owned core')
                        position = (rank[record.contig], record.start)
                        if previous_position is not None and position < previous_position: raise ValueError('Unsorted core VCF')
                        if position != previous_position: keys_at_position.clear()
                        key = tuple(record.alleles or ())
                        if key in keys_at_position: raise ValueError('Duplicate normalized variant at ' + str(position))
                        keys_at_position.add(key); previous_position = position
                        record.translate(header); out.write(record); count += 1
        if compressed: pysam.tabix_index(str(temporary), preset='vcf', force=True)
        os.replace(temporary, output)
        if compressed: os.replace(str(temporary) + '.tbi', str(output) + '.tbi')
    finally:
        temporary.unlink(missing_ok=True); Path(str(temporary) + '.tbi').unlink(missing_ok=True)
    return {'records': count, 'output_sha256': sha(output),
            'index_sha256': sha(str(output) + '.tbi') if compressed else None}


def build_parser():
    from prepare import add_inputs
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser, required=False)
    parser.add_argument('--regions', nargs='+')
    parser.add_argument('--contigs', nargs='+')
    parser.add_argument('--all-contigs', action='store_true')
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--trust-checkpoint', help='Use the trained v0.4 joint workflow')
    parser.add_argument('--config', help='v0.4 scientific configuration JSON')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--chunk-bp', type=int, default=4_000_000)
    parser.add_argument('--padding-bp', type=int, default=200_000)
    parser.add_argument('--cpu-threads',type=int,default=0,help='Total logical CPU budget; 0 uses affinity and cgroup limits')
    parser.add_argument('--jobs', type=int, default=None)
    parser.add_argument('--evidence-jobs',type=int,default=None,help='Processes per component; automatically allocated within the total CPU budget')
    parser.add_argument('--component-jobs',type=int,default=None,help='Independent read components per LD window; automatically allocated')
    parser.add_argument('--device', default='cpu'); parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--pairhmm-backend',choices=['cpu','cuda','auto'],default='cpu',help='Whole-read and partial-anchored likelihood batches; independent of neural --device')
    parser.add_argument('--cuda-precision',choices=['float64','mixed'],default='float64',help='Optional guarded float32 PairHMM workspace; log scales remain float64; requires validation')
    parser.add_argument('--compact-neural',action='store_true',help='Experimental inference-only omission of masked allele/molecule work')
    parser.add_argument('--neural-cpu-threads',type=int,default=1)
    parser.add_argument('--cuda-hap-bucket',type=int,choices=[32,128,512],default=128)
    parser.add_argument('--gpu-evidence-pipeline',action='store_true',help='Experimental CPU evidence actors feeding central CUDA batches')
    parser.add_argument('--cuda-kernel',choices=['pair','warp'],default='pair',help='Experimental row-parallel warp kernel or original independent-pair kernel')
    parser.add_argument('--cuda-device',type=int,default=0)
    parser.add_argument('--cuda-memory-mb',type=int,default=256,help='Maximum live PairHMM batch arrays per worker, excluding CUDA context/pool')
    parser.add_argument('--min-gp', type=float, default=0.); parser.add_argument('--min-gq', type=int, default=20)
    parser.add_argument('--variants-only', action='store_true'); parser.add_argument('--no-cohort', action='store_true')
    parser.add_argument('--plan-only', action='store_true', help='Show cores/padding without hashing BAMs or running inference')
    return parser


def main():
    global ARTIFACTS
    from prepare import validate, source_identity
    from base.input_identity import file_identity
    args = build_parser().parse_args(); started = time.monotonic()
    accelerator_runtime={'backend':args.pairhmm_backend}
    if args.cuda_device<0 or args.cuda_memory_mb<8:raise ValueError('Invalid CUDA device/memory budget')
    if args.evidence_jobs is not None and args.evidence_jobs<1:raise ValueError('evidence-jobs must be positive')
    os.environ['VARDEPTH_PAIRHMM_BACKEND']=args.pairhmm_backend
    os.environ['VARDEPTH_CUDA_DEVICE']=str(args.cuda_device)
    os.environ['VARDEPTH_CUDA_PRECISION']=args.cuda_precision
    if args.neural_cpu_threads<1:raise ValueError('Neural CPU threads must be positive')
    os.environ['VARDEPTH_COMPACT_NEURAL']='1' if args.compact_neural else '0'
    os.environ['VARDEPTH_NEURAL_CPU_THREADS']=str(args.neural_cpu_threads)
    os.environ['VARDEPTH_WARP_HAP_BUCKET']=str(args.cuda_hap_bucket)
    os.environ['VARDEPTH_CUDA_KERNEL']=args.cuda_kernel
    os.environ['VARDEPTH_GPU_EVIDENCE_PIPELINE']='1' if args.gpu_evidence_pipeline else '0'
    os.environ['VARDEPTH_CUDA_MEMORY_MB']=str(args.cuda_memory_mb)
    if args.pairhmm_backend!='cpu':
        Path(args.work_dir).mkdir(parents=True,exist_ok=True)
        os.environ['VARDEPTH_ACCELERATOR_LOG']=str(Path(args.work_dir).resolve()/'accelerator.jsonl')
        if not args.plan_only:
            from accelerator import CudaPairHMM
            try:accelerator_runtime['cuda']=CudaPairHMM(args.cuda_device,args.cuda_memory_mb).metadata
            except RuntimeError:
                if args.pairhmm_backend=='cuda':raise
                accelerator_runtime['fallback']='cpu'
    if args.trust_checkpoint:
        if not args.config: raise ValueError('--config is required with --trust-checkpoint')
        if args.min_gp != 0 or args.min_gq != 20 or args.variants_only:
            raise ValueError('v0.4 genome output retains all GP; confidence filtering is a separate operation')
        ARTIFACTS = ('padded.vcf.gz', 'padded.vcf.gz.tbi', 'padded.vcf.gz.pipeline.json',
                     'padded.vcf.gz.run.json', 'core.vcf.gz', 'core.vcf.gz.tbi')
        args.trust_checkpoint = str(Path(args.trust_checkpoint).resolve())
        args.config = str(Path(args.config).resolve())
    if not args.bam or not args.reference: raise ValueError('--bam and --reference are required')
    selection = [bool(args.region), bool(args.regions), bool(args.contigs), args.all_contigs]
    if sum(selection) != 1: raise ValueError('Choose exactly one of --region, --regions, --contigs, --all-contigs')
    if (args.jobs is not None and args.jobs<1) or args.batch_size<1 or (args.component_jobs is not None and args.component_jobs<1):raise ValueError('Worker counts and batch size must be positive')
    if not str(args.output).endswith(('.vcf', '.vcf.gz')): raise ValueError('Output must end in .vcf or .vcf.gz')
    if args.padding_bp < args.max_indel + args.context_span:
        raise ValueError('padding-bp must cover max-indel + context-span')
    if not 0 <= args.min_gp <= 1 or args.min_gq < 0: raise ValueError('Invalid confidence threshold')
    args.reference = str(Path(args.reference).resolve()); args.checkpoint = str(Path(args.checkpoint).resolve())
    lengths = reference_lengths(args.reference)
    if args.region: regions = [args.region]
    elif args.regions: regions = args.regions
    else:
        selected = list(lengths) if args.all_contigs else args.contigs
        if any(chrom not in lengths for chrom in selected): raise ValueError('Unknown reference contig')
        regions = [(chrom, 0, lengths[chrom]) for chrom in selected]
    chunks = plan_chunks(regions, lengths, args.chunk_bp, args.padding_bp)
    from cpu_schedule import available_cpus,allocate
    available=available_cpus()
    if args.cpu_threads<0 or args.cpu_threads>available:raise ValueError('CPU budget must fit process affinity and cgroup quota')
    budget=args.cpu_threads or available
    if args.neural_cpu_threads>budget:raise ValueError('Neural thread count exceeds CPU budget')
    cpu_plan=allocate(budget,len(chunks),max((chunk.core_end0-chunk.core_start0+999999)//1000000 for chunk in chunks),
                      args.jobs,args.component_jobs,args.evidence_jobs,args.neural_cpu_threads)
    args.jobs=cpu_plan['jobs'];args.component_jobs=cpu_plan['component_jobs'];args.evidence_jobs=cpu_plan['evidence_jobs']
    os.environ['VARDEPTH_EVIDENCE_JOBS']=str(args.evidence_jobs)
    os.environ['VARDEPTH_COMPONENT_CPU_BUDGET']=str(cpu_plan['component_budget'])
    serial_stages=cpu_plan['component_budget']==1 and args.component_jobs==1 and args.neural_cpu_threads==1 and args.device=='cpu'
    os.environ['VARDEPTH_INPROCESS_CPU_STAGES']='1' if serial_stages else '0'
    cpu_plan['serial_stages_share_process']=serial_stages
    validate(argparse.Namespace(**(vars(args) | {'region': chunks[0].region})))
    if args.plan_only:
        print(json.dumps({'version': VERSION, 'chunks': [asdict(chunk) for chunk in chunks],'cpu_schedule':cpu_plan}, indent=2)); return
    sample_ids = []; bam_files = []; canonical_bams = []
    for value in args.bam:
        if '=' not in value: raise ValueError('--bam must be SAMPLE=path')
        sample, path = value.split('=', 1)
        if not sample or any(c.isspace() for c in sample) or not path: raise ValueError('Invalid sample/BAM')
        path = str(Path(path).resolve()); canonical_bams.append(sample + '=' + path); bam_files.append(path)
        if sample not in sample_ids: sample_ids.append(sample)
    args.bam = canonical_bams
    for field in ('panel', 'genetic_map'):
        if getattr(args, field): setattr(args, field, str(Path(getattr(args, field)).resolve()))
    work = Path(args.work_dir).resolve(); work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('VARDEPTH_HASH_CACHE_DIR', str(work / 'input-hashes'))
    files = {args.reference, args.checkpoint, *bam_files}
    if args.trust_checkpoint: files.update([args.trust_checkpoint, args.config])
    files.update(getattr(args, field) for field in ('panel', 'genetic_map') if getattr(args, field))
    for path in list(files):
        for index in [path + '.fai', path + '.gzi', path + '.bai', path + '.crai', path + '.tbi', path + '.csi', str(Path(path).with_suffix('.bai'))]:
            if Path(index).is_file(): files.add(index)
    identities = {path: file_identity(path) for path in sorted(files)}
    sources = source_identity()
    native_receipt=HERE.parent/'native-python.json'
    if native_receipt.exists():
        native=json.loads(native_receipt.read_text())
        for relative,record in native['modules'].items():
            if sha(HERE.parent/relative)!=record['source_sha256'] or sha(HERE.parent/record['binary'])!=record['binary_sha256']:
                raise ValueError('Native Python build no longer matches its source; rebuild before inference')
        sources['native-python.json']=sha(native_receipt)
    for name in ('genome.py', 'pipeline.py', 'call.py', 'model.py'): sources[name] = sha(HERE / name)
    if args.trust_checkpoint:
        for name in ('run_v04.py', 'trust.py', 'fuse_trust.py', 'indel_refine.py', 'scaffold.py', 'scaffold.cpp', 'anchored.cpp', 'native_build.py'):
            sources[name] = sha(HERE / name)
        for name in ('run_v05.py','joint_ld.py','categorical.py','categorical.cpp','partial_anchor.py','triangular.py','projection.py','project.cpp','proposals.py'):
            sources[name] = sha(HERE / name)
    # Include every inference dependency in the cache identity, including new native kernels.
    for p in HERE.rglob('*'):
        if p.is_file() and p.suffix in ('.py','.pyx','.cpp','.cu','.sh','.hpp'):
            sources[str(p.relative_to(HERE))] = sha(p)
    ignored = {'region', 'regions', 'contigs', 'all_contigs', 'output', 'work_dir', 'jobs', 'plan_only'}
    semantics = {key: value for key, value in vars(args).items() if key not in ignored}
    from evidence_trace import enabled as trace_enabled, VERSION as TRACE_VERSION
    semantics['evidence_trace_version'] = TRACE_VERSION if trace_enabled() else None
    identity = dict(version=VERSION, settings=semantics, accelerator_runtime=accelerator_runtime,chunks=[asdict(chunk) for chunk in chunks],
                    inputs=identities, source_sha256=sources, samples=sample_ids)
    key = fingerprint(identity); run = work / ('run-' + key[:24]); run.mkdir(exist_ok=True)
    base_command = [sys.executable, str(HERE / ('run_v05.py' if args.trust_checkpoint else 'pipeline.py'))]
    for field in ('bam', 'reference', 'panel', 'genetic_map', 'exclude_samples', 'panel_reader', 'states', 'ne',
                  'min_support', 'min_bq', 'min_mq', 'max_indel', 'artifact_prior', 'indel_artifact_prior',
                  'context_gap', 'context_span', 'shard_rows', 'checkpoint', 'device', 'batch_size', 'min_gp', 'min_gq'):
        value = getattr(args, field)
        if args.trust_checkpoint and field in ('batch_size', 'min_gp', 'min_gq'): continue
        if value is None: continue
        for element in value if isinstance(value, list) else [value]:
            base_command.extend(['--' + field.replace('_', '-'), str(element)])
    if args.trust_checkpoint: base_command.extend(['--trust-checkpoint', args.trust_checkpoint, '--config', args.config, '--component-jobs',str(args.component_jobs)])
    if args.no_cohort: base_command.append('--no-cohort')
    if args.no_ld_proposals: base_command.append('--no-ld-proposals')
    if args.variants_only: base_command.append('--variants-only')
    with (run / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        atomic_json(run / 'identity.json', identity)
        try:
            receipts = schedule_chunks(chunks, lambda chunk: run_chunk(chunk, run / chunk.id, key, base_command,
                                                                      sample_ids, lengths), jobs=args.jobs)
            # Check every final chunk again before publishing any merged VCF.
            for chunk in chunks:
                if verified_completion(run / chunk.id, key, chunk, sample_ids, lengths) is None:
                    raise RuntimeError('Completed chunk changed before merge: ' + chunk.id)
            # Different run identities can share an output path. Serialize the
            # VCF/index/manifest publication to prevent interleaved generations.
            Path(args.output).resolve().parent.mkdir(parents=True, exist_ok=True)
            with Path(str(args.output) + '.genome.lock').open('a') as output_lock:
                fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX)
                if args.trust_checkpoint:
                    from ownership import reconcile
                    original = run / 'all-posterior-blocks.vcf.gz'
                    merge_owned_vcfs(chunks, [run / chunk.id / 'core.vcf.gz' for chunk in chunks], original, sample_ids, lengths)
                    from event_ownership import reconcile as reconcile_events
                    import uuid
                    unique = run / 'unique-posterior-blocks.vcf.gz'
                    selection = reconcile(original, args.reference, unique, overwrite=True)
                    temporary = run / (uuid.uuid4().hex + '.event-consensus.vcf.gz')
                    events = reconcile_events(unique, args.reference, temporary)
                    from snv_event_ownership import reconcile as reconcile_snvs
                    snv_temporary = run / (uuid.uuid4().hex + '.snv-consensus.vcf.gz')
                    snvs = reconcile_snvs(temporary, args.reference, snv_temporary)
                    os.replace(snv_temporary, args.output)
                    os.replace(str(snv_temporary) + '.tbi', str(args.output) + '.tbi')
                    os.replace(str(temporary) + '.event-ownership.json', str(args.output) + '.event-ownership.json')
                    os.replace(str(snv_temporary) + '.snv-event-ownership.json', str(args.output) + '.snv-event-ownership.json')
                    merged = dict(records=snvs['counts']['output_records'], output_sha256=snvs['output_sha256'],
                                  original_posterior_vcf=str(original), ownership=selection,
                                  event_input_vcf=str(unique), event_ownership=events,
                                  snv_event_input_vcf=str(temporary), snv_event_ownership=snvs)
                    if str(args.output).endswith('.gz'): merged['index_sha256'] = sha(str(args.output)+'.tbi')
                else:
                    merged = merge_owned_vcfs(chunks, [run / chunk.id / 'core.vcf.gz' for chunk in chunks],
                                             args.output, sample_ids, lengths)
                manifest = dict(identity=identity, run_key=key, chunks=receipts, jobs=args.jobs,cpu_schedule=cpu_plan,
                                output=str(Path(args.output).resolve()), **merged, seconds=time.monotonic() - started,
                                reused_chunks=sum(bool(r['reused']) for r in receipts), status='complete',
                                failure_policy='stop new submissions and drain already running jobs',
                                limitations=['Finite padded chunks approximate chromosome-wide LD/state selection and cohort context',
                                             'Normalized start0 ownership avoids seam duplicates but does not prove discovery completeness',
                                             'VCF, index and manifest are individually atomic; verify the manifest hashes before consuming',
                                             'No whole-genome accuracy or calibration claim follows from successful scheduling',
                                             'Overlap source selection preserves retained GP but does not merge counterfactual alternate catalogs; original posterior VCF is retained'],
                                completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                atomic_json(str(args.output) + '.genome.json', manifest)
            atomic_json(run / 'COMPLETE.json', manifest)
            (run / 'FAILED.json').unlink(missing_ok=True)
            print(json.dumps({key: manifest[key] for key in ('output', 'records', 'reused_chunks', 'seconds', 'status')}))
        except Exception as error:
            atomic_json(run / 'FAILED.json', dict(run_key=key, error=str(error), status='failed',
                                                  failure_policy='active workers drained; final output not published on chunk failure'))
            raise


if __name__ == '__main__':
    main()
