"""Whole-read semiglobal evidence for retained local candidate haplotypes.

Uses our existing independent PairHMM and diploid artifact marginalization.
Read windows are identical across hypotheses; there is no intersection of
candidate-specific internal anchors. This is an experimental dependent expert.
"""
import collections, math, sys
from pathlib import Path
import numpy as np
from whole_pairhmm import Prepared
from triangular import diploid_likelihood
from partial_anchor import genotype_exact_counts

def _inputs(row,inputs,fetch):
    reads=[]
    for bam,rg,fallback in inputs:
        for read in bam.fetch(row['chrom'],row['start0'],row['end0']):
            if read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate or read.is_qcfail or not 20<=read.mapping_quality<255 or read.query_qualities is None:continue
            sequence=read.query_sequence.upper();qualities=bytes(read.query_qualities)
            if len(sequence)<12 or (sum(qualities)/len(qualities))<20 or any(b not in 'ACGT' for b in sequence):continue
            lib=rg.get(read.get_tag('RG'),fallback) if read.has_tag('RG') else fallback
            reads.append((lib,read,sequence,qualities))
    padding=max([len(sequence)+50 for lib,read,sequence,qualities in reads]+[200])
    prefix=fetch(max(0,row['start0']-padding),row['start0']);suffix=fetch(row['end0'],row['end0']+padding)
    haps=[prefix+a+suffix for a in row['alleles']];fragments=collections.defaultdict(list)
    retained=[item for item in reads if item[1].reference_start>=row['start0']-padding and item[1].reference_end<=row['end0']+padding]
    return dict(reads=reads,retained=retained,haps=haps,padding=padding)

def _finish(row,context,likelihoods):
    n=len(row['alleles']);order=[(a,b) for b in range(n) for a in range(b+1)]
    reads=context['reads'];retained=context['retained'];haps=context['haps'];padding=context['padding']
    fragments=collections.defaultdict(list)
    from molecule_rows import normalized_rows
    batch_values,batch_certainty=normalized_rows(likelihoods,[item[1].mapping_quality for item in retained])
    for (lib,read,sequence,qualities),values,certainty in zip(retained,batch_values,batch_certainty):
        certainty=float(certainty)
        if certainty<math.log(3):continue
        matches={i for i,hap in enumerate(haps) if sequence in hap}
        fragments[lib,read.query_name].append((certainty,read.reference_start,read.reference_end,values,matches))
    molecules=[];compatible=[]
    for alignments in fragments.values():
        accepted=[]
        for item in sorted(alignments,key=lambda v:-v[0]):
            if any(max(item[1],old[1])<min(item[2],old[2]) for old in accepted):continue
            accepted.append(item)
        logl=sum((np.log(np.maximum(1e-300,x[3])) for x in accepted),np.zeros(n))
        molecules.append(np.exp(logl-logl.max()));compatible.append(set.intersection(*(x[4] for x in accepted)))
    matrix=np.array(molecules) if molecules else np.empty((0,n))
    gl=diploid_likelihood(matrix,.001)
    # An explicit uniform haplotype prior isolates this new read expert.
    prior=np.array([1 if a==b else 2 for a,b in order],float);prior/=prior.sum()
    lp=gl+np.log(prior);gp=np.exp(lp-lp.max());gp/=gp.sum()
    counts=genotype_exact_counts(compatible,n);best=int(gp.argmax());a,b=order[best];x,y,both=counts[best]
    support=x+y+both;minimum=x if a==b else min(x,y);exact=np.zeros(n,int)
    for matches in compatible:
        if len(matches)==1:exact[next(iter(matches))]+=1
    answer=dict(exact_ad=exact.tolist(),trusted=bool(len(molecules)>=6 and gp[best]>=.9 and minimum>=2 and support>=.7*len(molecules)),support_fraction=support/max(1,len(molecules)),partial_molecules=0,proposed_sequences={},dp=len(molecules),gp=gp.tolist(),pl=(-10/math.log(10)*(gl-gl.max())).tolist(),
        exact_genotype_counts=counts,input_alignments=len(reads),padding=padding,
        scope='Whole-read semiglobal PairHMM; overlapping mates counted once; uniform haplotype prior; not calibrated fusion')
    from evidence_trace import enabled
    if enabled():answer['_lineage_keys']=list(fragments)
    return answer

def _attach_lineage(answer,sample):
    from evidence_trace import molecules
    if '_lineage_keys' in answer:
        answer['lineage_molecules']=molecules(sample,answer.pop('_lineage_keys'))
        answer['lineage_kind']='whole_read_uniform_prior'
    return answer

def score(row,bams,fetch):
    result={}
    for sample,inputs in bams.items():
        context=_inputs(row,inputs,fetch);retained=context['retained']
        likelihoods=Prepared(context['haps'],mode=4).many([x[2] for x in retained],[x[3] for x in retained])
        result[sample]=_attach_lineage(_finish(row,context,likelihoods),sample)
    return result

def score_rows(rows,bams,fetch_for_chrom,max_pairs=32768):
    """Bounded across-locus CUDA batches; restore row and sample order."""
    from accelerator import backend,score_pairs
    if backend()=='cpu':return [score(row,bams,fetch_for_chrom(row['chrom'])) for row in rows]
    results=[{} for row in rows];contexts=[];jobs=[]
    def flush():
        if not contexts:return
        likelihoods=score_pairs(jobs);offset=0
        for i,sample,context in contexts:
            shape=(len(context['retained']),len(context['haps']));n=shape[0]*shape[1]
            results[i][sample]=_attach_lineage(_finish(rows[i],context,likelihoods[offset:offset+n].reshape(shape)),sample);offset+=n
        contexts.clear();jobs.clear()
    for i,row in enumerate(rows):
        for sample,inputs in bams.items():
            context=_inputs(row,inputs,fetch_for_chrom(row['chrom']))
            count=len(context['retained'])*len(context['haps'])
            if jobs and len(jobs)+count>max_pairs:flush()
            contexts.append((i,sample,context))
            jobs.extend((x[2],x[3],h) for x in context['retained'] for h in context['haps'])
            if len(jobs)>=max_pairs:flush()
    flush();return results
