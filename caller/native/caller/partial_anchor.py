"""Repeat-safe anchored likelihoods from full and partial local read windows.

The retained candidate catalog is unchanged. A partial observation scores only
its visible candidate sequence; alleles identical in that window have exactly
equal likelihood. Overlapping mates are counted once per physical molecule.
"""
import argparse,collections,concurrent.futures as cf,ctypes,hashlib,json,math
from pathlib import Path
import sys,time
import numpy as np
import pysam
from projection import common_anchors,crop_observation,expand_compound_context

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'base'))
from indel_refine import kernel
from triangular import diploid_likelihood


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def genotype_exact_counts(compatible,n):
    """Count exact compatibility once per molecule for each diploid hypothesis.

    A read can match several retained haplotypes. It can still distinguish the
    two alleles of a particular genotype. Reads matching both alleles supply
    union support but are excluded from heterozygous balance evidence.
    """
    hits=np.zeros((len(compatible),n),np.int32)
    for i,possible in enumerate(compatible):
        for a in possible:hits[i,a]=1
    marginal=hits.sum(0);intersection=hits.T@hits
    return [[int(marginal[a]),0,0] if a==b else
            [int(marginal[a]-intersection[a,b]),int(marginal[b]-intersection[a,b]),int(intersection[a,b])]
            for b in range(n) for a in range(b+1)]


def _refine_steps(row,original,bams,fetch,lib,*,force_cpu=False):
    input_row=row
    new_start,new_alleles=expand_compound_context(row['start0'],row['alleles'],fetch)
    row=dict(row,alleles=new_alleles,start0=new_start,end0=new_start+len(new_alleles[0]),
             original_refinement_start0=row['start0'],original_refinement_end0=row['end0'])
    start=row['start0']-3;end=row['end0']+3
    prefix=fetch(start,row['start0']);suffix=fetch(row['end0'],end)
    alleles=[prefix+a+suffix for a in row['alleles']];reference=fetch(start,end)
    if alleles[0]!=reference or len(prefix)!=3 or len(suffix)!=3:
        return row,dict(fallback='reference boundary')
    maps,safe=common_anchors(start,reference,alleles,fetch)
    n=len(alleles);order=[(a,b) for b in range(n) for a in range(b+1)]
    af=np.array([max(.001,float(v or 0)) for v in original['panel_af']]+[.001]*row['added_alleles'])
    af/=af.sum();prior=np.array([af[a]*af[b]*(1 if a==b else 2) for a,b in order]);prior/=prior.sum()
    result=dict(row);result['samples']={};diagnostics={}
    for sample,inputs in bams.items():
        fragments=collections.defaultdict(list);seen=0;partial=0;pending=[]
        for bam,rg,fallback in inputs:
            for read in bam.fetch(row['chrom'],max(0,start),end):
                if read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate or read.is_qcfail or not 10<=read.mapping_quality<255 or read.query_qualities is None:continue
                seen+=1
                o=crop_observation(read,start,reference,alleles,maps,safe)
                if o is None or not o['qualities'] or (sum(o['qualities'])/len(o['qualities']))<20 or any(b not in 'ACGT' for b in o['sequence']):continue
                library=rg.get(read.get_tag('RG'),fallback) if read.has_tag('RG') else fallback
                pending.append((read,o,library))
        from accelerator import backend,score_pairs
        gpu_likelihoods=None
        if backend()!='cpu' and pending and not force_cpu:
            jobs=[(o['sequence'],o['qualities'],hap) for read,o,library in pending for hap in sorted(set(o['haplotypes']))]
            gpu_likelihoods=iter((yield jobs))
        elif pending and (not force_cpu or (sys.platform=='linux' and __import__('platform').machine() in ('x86_64','AMD64'))):
            from cpu_batch import pairs
            jobs=[(o['sequence'],o['qualities'],hap) for read,o,library in pending for hap in sorted(set(o['haplotypes']))]
            # Canonical mode retains the original per-row scaling, operation
            # order and log accumulation. It does not use prefix/suffix reuse.
            gpu_likelihoods=iter(pairs(jobs,mode=1,canonical=force_cpu))
        batch_values=batch_certainty=None
        if gpu_likelihoods is not None:
            from molecule_rows import normalized_rows
            likelihood_rows=[]
            for read,o,library in pending:
                likelihood={hap:next(gpu_likelihoods) for hap in sorted(set(o['haplotypes']))}
                likelihood_rows.append([likelihood[h] for h in o['haplotypes']])
            batch_values,batch_certainty=normalized_rows(likelihood_rows,[read.mapping_quality for read,o,library in pending])
        for row_index,(read,o,library) in enumerate(pending):
                if gpu_likelihoods is None:
                    quality=(ctypes.c_ubyte*len(o['qualities']))(*o['qualities']);sequence=o['sequence'].encode()
                # Cache equal visible candidate strings before expensive DP.
                if batch_values is None:
                    likelihood={}
                    for hap in sorted(set(o['haplotypes'])):
                        likelihood[hap]=lib.anchored_pairhmm(sequence,quality,len(sequence),hap.encode(),len(hap),.001,.1)
                    ll=np.array([likelihood[h] for h in o['haplotypes']]);values=np.exp(ll-ll.max())
                    error=10**(-min(read.mapping_quality,60)/10);values=(1-error)*values+error*values.mean()
                    certainty=float(np.ptp(np.log(np.maximum(1e-300,values))))
                else:values=batch_values[row_index];certainty=float(batch_certainty[row_index])
                if certainty<math.log(3):continue
                matches={i for i,h in enumerate(o['haplotypes']) if h==o['sequence']}
                fragments[library,read.query_name].append((certainty,o['left'],o['right'],values,matches,o['whole_context']))
        molecules=[];exact=np.zeros(n,int);compatible=[]
        for key,alignments in fragments.items():
            accepted=[]
            for item in sorted(alignments,key=lambda v:-v[0]):
                if any(max(item[1],old[1])<min(item[2],old[2]) for old in accepted):continue
                accepted.append(item)
            logl=sum((np.log(np.maximum(1e-300,x[3])) for x in accepted),np.zeros(n))
            molecules.append(np.exp(logl-logl.max()))
            matches=set.intersection(*(x[4] for x in accepted))
            compatible.append(matches)
            if len(matches)==1:exact[next(iter(matches))]+=1
            partial+=int(any(not x[5] for x in accepted))
        matrix=np.array(molecules) if molecules else np.empty((0,n))
        if force_cpu:
            from triangular import _diploid_likelihood_numpy
            gl=_diploid_likelihood_numpy(matrix,.001)
        else:gl=diploid_likelihood(matrix,.001)
        lp=gl+np.log(prior)
        # A tiny DP difference can flip argmin(PL), a discrete fusion feature.
        # Recover the training CPU convention before such a discontinuity.
        # This is a numerical guard, independent of coordinates/truth labels.
        if gpu_likelihoods is not None and not force_cpu and len(gl)>1:
            margin=1e-4*max(1,len(molecules))
            gaps=[float(np.diff(np.partition(values,-2)[-2:])[0])
                  for values in (gl,lp)]
            if min(gaps)<=margin:
                canonical,detail=yield from _refine_steps(input_row,original,{sample:inputs},fetch,lib,force_cpu=True)
                result['samples'][sample]=canonical['samples'][sample]
                diagnostics[sample]=dict(detail[sample],numerical_guard=dict(
                    backend='cpu_float64',reason='near_tied_likelihood_or_posterior',
                    log_likelihood_margin=margin,observed_gaps=gaps))
                continue
        gp=np.exp(lp-lp.max());gp/=gp.sum();best=int(gp.argmax());selected=set(order[best])
        support=sum(bool(matches&selected) for matches in compatible)
        counts=genotype_exact_counts(compatible,n);only_a,only_b,both=counts[best]
        minimum=only_a if len(selected)==1 else min(only_a,only_b)
        trusted=len(molecules)>=6 and gp[best]>=.9 and minimum>=2 and support>=.7*len(molecules)
        result['samples'][sample]=dict(gp=gp.tolist(),pl=(-10/math.log(10)*(gl-gl.max())).tolist(),
            dp=len(molecules),exact_ad=exact.tolist(),trusted=bool(trusted),support_fraction=support/max(1,len(molecules)),
            proposed_sequences=row['samples'][sample].get('proposed_sequences',{}),
            partial_molecules=partial,exact_ad_semantics='Unique exact match across accepted visible sequence windows; ambiguous exact matches not allocated',
            original_spanning_dp=row['samples'][sample]['dp'])
        result['samples'][sample]['exact_genotype_counts']=counts
        from evidence_trace import enabled, molecules as trace_molecules
        if enabled():
            result['samples'][sample]['lineage_molecules']=trace_molecules(sample,fragments.keys())
            result['samples'][sample]['lineage_kind']='partial_anchored'
        result['samples'][sample]['exact_genotype_count_semantics']='A-only,B-only,both compatibility counts per VCF-order genotype; both-match reads do not support heterozygous balance; homozygotes use A-only'
        diagnostics[sample]=dict(fetched_alignments=seen,informative_molecules=len(molecules),partial_molecules=partial,
            original_spanning_dp=row['samples'][sample]['dp'],new_best_GT=list(order[best]))
    result['partial_anchored']=True
    return result,diagnostics


def refine(row,original,bams,fetch,lib):
    """Retain the single-locus API for compatibility and CPU inference."""
    from accelerator import score_pairs
    generator=_refine_steps(row,original,bams,fetch,lib)
    try:
        jobs=next(generator)
        while True:jobs=generator.send(score_pairs(jobs,anchored=True))
    except StopIteration as done:return done.value


def refine_rows(rows,original,bams,fetch_for_chrom,lib,max_pairs=32768):
    """Batch independent anchored likelihoods across loci and samples.

    The coroutine retains the original per-molecule calculation, posterior and
    exact-support gates. Only independent DP jobs change execution order.
    Pending batches are bounded, apart from a single locus that is subsequently
    split according to the accelerator's explicit device-memory budget.
    """
    from accelerator import score_pairs
    from batch_coroutines import score_generators
    generators=(_refine_steps(row,original[row['record_id']],bams,fetch_for_chrom(row['chrom']),lib) for row in rows)
    return score_generators(generators,lambda jobs:score_pairs(jobs,anchored=True),max_pairs)
