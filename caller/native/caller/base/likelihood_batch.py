"""Independent batched evaluation of already selected indel observations.

Preparation is sequential: FASTA handles are never shared across threads.
Only independent DP jobs are regrouped. Results return in original input order.
"""
import math
import numpy as np

def indel_inputs(v,reads,fetch,min_bq):
    from alignment_queries import window_bounds
    pos,ref,alt=v;left=max(0,pos-100);right=pos+len(ref)+101
    rh=fetch(left,right).upper();ah=rh[:pos-left]+alt+rh[pos-left+len(ref):];pending=[]
    for library,r in (reads.at(pos) if hasattr(reads,'at') else reads):
        if r.reference_end<=pos or r.reference_start>pos:continue
        if r.reference_start>pos-3 or r.reference_end<pos+len(ref)+3:continue
        bounds=window_bounds(r,left,right)
        if bounds is None:continue
        q0,q1=bounds
        if q1-q0<12:continue
        seq=r.query_sequence[q0:q1].upper();q=bytes(r.query_qualities[q0:q1])
        if not q or sum(q)/len(q)<min_bq:continue
        pending.append((library,r,seq,q))
    return pending,[rh,ah]

def finish_indel(pending,values,*,compute_gl=True):
    per_fragment={}
    for (library,r,seq,q),(a,b) in zip(pending,values):
        s=max(a,b);a,b=math.exp(a-s),math.exp(b-s);mq=10**(-min(r.mapping_quality,60)/10)
        mix=(a+b)/2;lr=(1-mq)*a+mq*mix;la=(1-mq)*b+mq*mix
        key=(library,r.query_name)
        val=(np.log([max(1e-300,lr),max(1e-300,(lr+la)/2),max(1e-300,la)]) if compute_gl else None,abs(math.log(max(lr,1e-300)/max(la,1e-300))),lr,la)
        if key not in per_fragment or val[1]>per_fragment[key][1]:per_fragment[key]=val
    return per_fragment

def indel_batches(variants,reads,fetch,min_bq,max_pairs=32768):
    from accelerator import score_pairs
    result={};pending_blocks=[];jobs=[]
    def flush():
        if not pending_blocks:return
        likelihood=score_pairs(jobs);offset=0
        for i,observations,n in pending_blocks:
            result[i]=finish_indel(observations,likelihood[offset:offset+n].reshape(len(observations),2));offset+=n
        pending_blocks.clear();jobs.clear()
    for i,v in enumerate(variants):
        if len(v[1])==len(v[2])==1:continue
        observations,haps=indel_inputs(v,reads,fetch,min_bq)
        if jobs and len(jobs)+2*len(observations)>max_pairs:flush()
        block_jobs=[(s,q,h) for library,r,s,q in observations for h in haps]
        jobs.extend(block_jobs);pending_blocks.append((i,observations,len(block_jobs)))
        if len(jobs)>=max_pairs:flush()
    flush();return result
