"""Own v0.1.6 algorithm exposed as a truth-free, external-only feature stage."""
from __future__ import annotations
from reference_cache import CachedReference

import collections
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / 'base'


def base_modules():
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))
    import caller
    if Path(caller.__file__).resolve().parent != BASE:
        raise RuntimeError('A different caller module was imported into this process')
    return caller


def parse_region(region):
    try:
        chrom, bounds = region.rsplit(':', 1)
        start, end = map(int, bounds.split('-'))
    except (ValueError, AttributeError):
        raise ValueError('Region must be CHROM:START-END, 1-based inclusive') from None
    if start < 1 or end < start:
        raise ValueError('Invalid or empty region')
    return chrom, start - 1, end


def unsupported_reference_alleles(start, alleles, fetch):
    """Report inherited IUPAC context the ACGTN model cannot represent.

    Leave empty/symbolic/malformed alleles to the existing strict normalizer.
    Never replace an ambiguous reference base by a guessed nucleotide.
    """
    if not alleles or any(not a for a in alleles):
        return None
    invalid = set(''.join(alleles).upper()) - set('ACGTN')
    reference_invalid = set(alleles[0].upper()) - set('ACGTN')
    if (invalid and invalid <= set('RYSWKMBDHV') and reference_invalid
            and fetch(start, start + len(alleles[0])).upper() == alleles[0].upper()):
        return dict(start0=start, end0=start + len(alleles[0]),
                    reference=alleles[0], unsupported_symbols=sorted(invalid),
                    reason='Inherited IUPAC reference context unsupported by ACGTN local allele model')
    return None


def compute(args):
    """One external-only HMM pass for each sample, shared truth-free catalog.

    No posterior recycling or legacy cohort pass. Neural cohort conditioning is
    constructed later from these independent sample estimates by call.py.
    """
    from features import normalize_record
    caller = base_modules()
    from read_index import ReadIndex
    from context_blocks import make_context_blocks
    from local_alleles import supported_candidates, sequence_likelihoods, sequence_likelihoods_many
    from artifact import site_likelihoods
    import pysam

    begun = time.monotonic()
    chrom, start, end = parse_region(args.region)
    fa = CachedReference(args.reference)
    if chrom not in fa.references or end > fa.get_reference_length(chrom):
        raise ValueError('Region outside FASTA contig bounds')
    fetch = lambda a, b: fa.fetch(chrom, max(0, a), min(fa.get_reference_length(chrom), b))
    bams = collections.OrderedDict()
    for value in args.bam:
        if '=' not in value:
            raise ValueError('--bam requires SAMPLE=path')
        sample, path = value.split('=', 1)
        if not sample or any(c.isspace() for c in sample) or not path:
            raise ValueError('Invalid sample name or BAM path')
        bams.setdefault(sample, []).append(path)
    seen_paths, seen_sms = set(), set()
    excluded = set(filter(None, args.exclude_samples.split(','))) | set(bams)
    raw, headers, observations, initial, sequence_data = {}, {}, {}, {}, {}
    observation_alignments={}
    sources = collections.Counter(); fragment_seeds = collections.defaultdict(set)
    for sample, paths in bams.items():
        resolved = {str(Path(p).resolve()) for p in paths}
        if seen_paths & resolved:
            raise ValueError('Same BAM assigned to multiple people')
        seen_paths |= resolved
        raw[sample], sms, headers[sample] = caller.collect_reads(paths, args.reference, chrom, start, end, args.min_mq)
        platforms = {str(r['PL']).upper() for h in headers[sample] for r in h.get('RG',[]) if r.get('PL')}
        if platforms - {'ILLUMINA'}:
            raise ValueError('Current neural feature preset supports Illumina reads only; observed PL: '+','.join(sorted(platforms)))
        if seen_sms & sms:
            raise ValueError('Overlapping BAM SM identifiers: possible duplicate individual')
        seen_sms |= sms; excluded |= sms
        support = caller.discover(raw[sample], fetch, start, end, args.min_bq, args.max_indel)
        for variant, fragments in support.items():
            sources[variant] += len(fragments)
            fragment_seeds[variant].update((sample, *key) for key in fragments)
    for family in caller.FAMILIES:
        if family & excluded:
            excluded |= family
    reader = caller.load_panel
    if args.panel_reader == 'fast':
        from panel_fast import load_panel_fast
        reader = load_panel_fast
    panel, panel_names, panel_skips = reader(args.panel, chrom, start, end, fetch, excluded, args.max_indel,
        **({"reference_identity":args.reference} if args.panel_reader=="fast" else {}))
    actual_exclusions = []
    if args.panel:
        with pysam.VariantFile(args.panel) as vcf:
            actual_exclusions = sorted(set(vcf.header.samples) & excluded)
    read_sources = {v: n for v, n in sources.items() if n >= args.min_support}
    variants, rejected = supported_candidates(sorted(set(panel) | set(read_sources)), read_sources, panel, 12)
    catalog_cache={}
    blocks = make_context_blocks(variants, fetch, catalog_cache=catalog_cache, context_gap=args.context_gap,
                                 context_span=args.context_span, partial_reads=True,
                                 max_distance=500, panel=panel, support=read_sources,
                                 fragment_seeds=fragment_seeds, max_alleles=12)
    lib = caller.core()
    factor_caches={sample:{} for sample in raw}
    for sample, reads in raw.items():
        observation_alignments[sample]={}
        if not variants:
            observations[sample] = (np.zeros((0, 3)), np.zeros(0, int), np.zeros((0, 2), int), np.zeros(0, int), {})
            initial[sample] = (np.zeros((0, 3)), np.zeros((0, 2)), {'states': 0, 'empty_interval': True})
            sequence_data[sample] = {}
            continue
        print(f'{sample}: external-only base inference, {len(variants)} candidates', flush=True)
        indexed = ReadIndex(reads)
        obs = caller.likelihoods(variants, indexed, fetch, args.min_bq, lib,alignments=observation_alignments[sample],compute_gl=False)
        observations[sample] = (site_likelihoods(variants, obs[4], args.artifact_prior, args.indel_artifact_prior), *obs[1:])
        indices=[t for t,block in enumerate(blocks) if block['kind']=='multiallelic']
        sequence_data[sample]=dict(zip(indices,sequence_likelihoods_many([blocks[t] for t in indices],indexed,fetch,args.min_bq,lib)))
        initial[sample] = caller.infer(lib, variants, panel, observations[sample][0], args.states,
                                      fragment_observations=obs[4], artifact_prior=args.artifact_prior,
                                      state_selection='paired', local_blocks=blocks,
                                      sequence_data=sequence_data[sample], genetic_map=args.genetic_map,
                                      ne=args.ne, chrom=chrom, indel_artifact_prior=args.indel_artifact_prior, factor_cache=factor_caches[sample])
    ld_stage={'enabled':not getattr(args,'no_ld_proposals',False),'proposed_blocks':0,'affected_blocks':0,
              'promoted_sequences':0,'rerun_samples':0,
              'policy':'Two posterior-LD external haplotype proposals per shared block; preserve all existing read-supported sequences; recompute likelihood and rerun from original panel/read-only GL; never multiply initial GP'}
    if variants and ld_stage['enabled']:
        from ld_proposals import select_proposals,changed_blocks,preserve_read_supported_alleles
        proposals=select_proposals(initial,blocks)
        revised=make_context_blocks(variants,fetch,catalog_cache=catalog_cache,context_gap=args.context_gap,context_span=args.context_span,
                                    partial_reads=True,max_distance=500,panel=panel,support=read_sources,
                                    fragment_seeds=fragment_seeds,max_alleles=12,proposal_vectors=proposals)
        unguarded=changed_blocks(blocks,revised)
        revised=preserve_read_supported_alleles(blocks,revised,variants,read_sources)
        changed=changed_blocks(blocks,revised)
        ld_stage.update(proposed_blocks=len(proposals),affected_blocks=len(changed),
                        read_supported_guarded_blocks=len(unguarded)-len(changed),
                        promoted_sequences=sum(len(set(revised[t]['alleles'])-set(blocks[t]['alleles'])) for t in changed),
                        affected_first_seed_positions=[variants[blocks[t]['ids'][0]][0] for t in changed])
        if changed:
            print(f'LD-first proposals changed {len(changed)} local blocks; recomputing their molecular likelihoods',flush=True)
            blocks=revised
            for sample,reads in raw.items():
                indexed=ReadIndex(reads)
                sequence_data[sample].update(zip(changed,sequence_likelihoods_many([blocks[t] for t in changed],indexed,fetch,args.min_bq,lib)))
                initial[sample]=caller.infer(lib,variants,panel,observations[sample][0],args.states,
                                             fragment_observations=observations[sample][4],artifact_prior=args.artifact_prior,
                                             state_selection='paired',local_blocks=blocks,sequence_data=sequence_data[sample],
                                             genetic_map=args.genetic_map,ne=args.ne,chrom=chrom,
                                             indel_artifact_prior=args.indel_artifact_prior,factor_cache=factor_caches[sample],
                                             selected_states=initial[sample][2]['selected_external_haplotypes'])
            ld_stage['rerun_samples']=len(raw)
    factor_caches.clear();catalog_cache.clear()
    joint = next(iter(initial.values()))[2].get('joint_records', {})
    records, skipped, used_keys = [], set(), set()
    unsupported_records = []
    for i, variant in enumerate(variants):
        if i in skipped:
            continue
        if i in joint:
            data = joint[i]; ids = data['ids']; skipped.update(ids[1:])
            position, alleles = data['start'], tuple(data['alleles'])
            af = data.get('allele_af') or [None] * len(alleles)
        else:
            ids = [i]; position, ref, alt = variant; alleles = (ref, alt)
            af = [1 - panel[variant][1], panel[variant][1]] if variant in panel else [None, None]
        original_start = position; original_end = position + len(alleles[0])
        unsupported = unsupported_reference_alleles(position, alleles, fetch)
        if unsupported is not None:
            unsupported_records.append(dict(unsupported, seed_indices=list(ids), allele_count=len(alleles)))
            continue
        position, alleles = normalize_record(position, alleles, fetch)
        rid = f'{chrom}:{position+1}:{alleles[0]}:{",".join(alleles[1:])}'
        if rid in used_keys:
            raise ValueError('Duplicate normalized local record: ' + rid)
        used_keys.add(rid)
        from_panel = any(variants[j] in panel for j in ids)
        from_reads = any(variants[j] in read_sources for j in ids)
        records.append(dict(record_id=rid, chrom=chrom, start0=position, end0=position + len(alleles[0]),
                            alleles=list(alleles), evidence_start0=original_start, evidence_end0=original_end,
                            seed_indices=ids, seed_variants=[dict(start0=variants[j][0], ref=variants[j][1], alt=variants[j][2]) for j in ids],
                            source='BOTH' if from_panel and from_reads else 'PANEL' if from_panel else 'READS',
                            panel_af=[float(x) if x is not None else None for x in af],
                            read_support=int(sum(read_sources.get(variants[j], 0) for j in ids)),
                            joint=i in joint, first_seed=i))
    records.sort(key=lambda r: (r['start0'], tuple(r['alleles'])))
    for i, record in enumerate(records):
        record['record_index'] = i
    metadata_warnings = []
    for sample, items in headers.items():
        if any(not h.get('RG') or any(not r.get('LB') for r in h.get('RG', [])) for h in items):
            metadata_warnings.append(sample + ': missing library metadata uses file plus RG namespace; BAM/CRAM conversion requires preserving LB to retain cross-file identity')
        if any(not h.get('RG') or any(not r.get('PL') for r in h.get('RG',[])) for h in items):
            metadata_warnings.append(sample + ': PL absent in one or more read groups; Illumina platform cannot be verified from header')
    return dict(samples=list(bams), bams=bams, raw=raw, variants=variants, panel=panel, panel_names=panel_names,
                records=records, observations=observations, initial=initial,observation_alignments=observation_alignments,
                sequence_data=sequence_data, blocks=blocks, fasta=fa, fetch=fetch,
                block_by_seed={b['ids'][0]: t for t, b in enumerate(blocks)},
                manifest=dict(chrom=chrom, start0=start, end0=end, contig_length=fa.get_reference_length(chrom),
                              base_wall_seconds=time.monotonic() - begun, candidates=len(variants),
                              records=len(records), rejected_candidates=len(rejected),
                              unsupported_reference_records=unsupported_records,
                              panel_samples=len(panel_names), panel_skips=panel_skips,
                              actual_panel_exclusions=actual_exclusions,
                              excluded_panel_samples=sorted(excluded), metadata_warnings=metadata_warnings,
                              selected_external_haplotypes={s: initial[s][2].get('selected_external_haplotypes', []) for s in bams},
                              ld_proposal_stage=ld_stage,
                              inference_pass='external-only; no cohort posterior fed into donors',
                              local_alignment='LD posterior selected local sequence proposals followed by affected-block PairHMM realignment; no full personalized-reference remap or pangenome placement'))


def fragment_lookup(reads):
    result = collections.defaultdict(list)
    for library, read in reads:
        result[(library, read.query_name)].append(read)
    return result


def initial_cohort_context(base, enabled=True):
    """Automatic training/inference context from initial external-only estimates."""
    s, n = len(base['samples']), len(base['records'])
    if not enabled or s == 1 or n == 0:
        return np.zeros((s,n,12,6),np.float32), np.zeros((s,n),np.float32), {
            'enabled':bool(enabled), 'reason':'explicit ablation' if not enabled else 'single sample or empty catalog',
            'initial_estimates':'external-only; shared catalog', 'samples':s}
    from cohort_context import build_cohort_context
    gp = np.zeros((s,n,78),np.float32); pl = np.zeros_like(gp); depth = np.zeros((s,n),np.float32)
    for record in base['records']:
        j = record['record_index']; i = record['first_seed']; count = len(record['alleles']); ng = count*(count+1)//2
        for k,sample in enumerate(base['samples']):
            if record['joint']:
                payload = base['initial'][sample][2]['joint_records'][i]
                post, logl, dp = payload['gp'],payload['pl'],payload['dp']
            else:
                post = base['initial'][sample][0][i]
                logl = base['observations'][sample][0][i]; dp = base['observations'][sample][1][i]
            gp[k,j,:ng] = post
            pl[k,j,:ng] = np.minimum(999.,-10/np.log(10)*(logl-np.max(logl)))
            depth[k,j] = dp
    result = build_cohort_context(gp,np.array([len(r['alleles']) for r in base['records']]),
                                  np.array([r['start0'] for r in base['records']]),base['samples'],read_depth=depth,
                                  contigs=np.array([r['chrom'] for r in base['records']]),read_pl=pl)
    from evidence_trace import enabled as trace_enabled
    if trace_enabled():
        base['_lineage_cohort']=dict(samples=base['samples'],donor_indices=result.donor_indices,
            donor_weights=result.donor_weights,flank_indices=result.provenance.get('lineage_flank_indices',{}))
    return result.allele_context,result.context_strength,{'enabled':True,**result.provenance}


def record_evidence(base, record, sample, raw_fragments):
    """Return aligned genotype and molecular features for one actual VCF row."""
    from features import alignment_metadata, combine_alignment_metadata, DeferredMetadata
    from alignment_queries import cached_point_query
    from artifact import observation_probabilities
    metadata_cache=base.setdefault('_metadata_cache',{})
    query_cache=base.setdefault('_query_cache',{})
    i = record['first_seed']
    if record['joint']:
        payload = base['initial'][sample][2]['joint_records'][i]
        seq = base['sequence_data'][sample][base['block_by_seed'][i]]
        meta = DeferredMetadata(len(seq['fragment_keys']),lambda i:combine_alignment_metadata(seq['fragment_alignments'][i],cache=metadata_cache))
        return payload, seq['lh'], seq['fragment_keys'], meta
    pos, ref, alt = base['variants'][i]
    if len(ref) != 1 or len(alt) != 1:
        raise RuntimeError('Non-SNV record without a joint local allele likelihood')
    gl, dp, ad, _, observations = base['observations'][sample]
    payload = dict(gp=base['initial'][sample][0][i], pl=gl[i], dp=dp[i], ad=ad[i])
    keys, likelihood, metadata = [], [], []
    for key, (base_nt, bq, mq) in observations.get(i, {}).items():
        likelihood.append(observation_probabilities(min(bq,60),min(mq,60),base_nt==ref,base_nt==alt))
        keys.append(key); metadata.append((key,base_nt,bq,mq))
    def selected_metadata(index):
        key,base_nt,bq,mq=metadata[index]
        retained=base.get('observation_alignments',{}).get(sample,{}).get(i,{}).get(key)
        if retained is not None:r,q=retained
        else:
            candidates=[]
            for r in raw_fragments[key]:
                if r.query_qualities is None:continue
                q=cached_point_query(r,pos,query_cache)
                if q is not None and r.query_qualities[q]==bq and r.mapping_quality==mq and r.query_sequence[q].upper()==base_nt:
                    candidates.append((r,q))
            if not candidates:raise RuntimeError('Original molecular SNV observation cannot be matched to its BAM alignment')
            r,q=candidates[0]
        return alignment_metadata(r,q,q+1,1.,cache=metadata_cache)
    return payload, np.array(likelihood, dtype=np.float64).reshape(-1, 2), keys, DeferredMetadata(len(keys),selected_metadata)
