"""Compatible local multiallelic blocks for SNVs and short indels.

Overlapping candidates become alternative local sequences. Diploid combinations
are inferred jointly, not as independent binary calls. This is bounded seeded
haplotype enumeration, not de novo assembly from soft-clipped reads.
"""
import collections,ctypes,itertools,math
from functools import lru_cache
from collections import OrderedDict
import numpy as np
from row_patterns import unique_rows
from artifact import BALANCE,BALANCE_WEIGHT,CONTAMINATION,CONTAMINATION_WEIGHT,prior_for

def overlap_groups(variants):
    groups=[];group=[];end=-1
    for j,v in enumerate(variants):
        if group and v[0]>=end:groups.append(group);group=[]
        if not group:end=-1
        group.append(j);end=max(end,v[0]+len(v[1]))
    if group:groups.append(group)
    return groups

def supported_candidates(variants,support,panel,max_alleles=12,max_span=100):
    """Bound complexity by evidence, never truth. Return selected/rejected sets."""
    selected=[];rejected=[]
    for ids in overlap_groups(variants):
        cluster=[variants[j] for j in ids]
        if max(v[0]+len(v[1]) for v in cluster)-cluster[0][0]>max_span:
            # Complexity pruning must not discard directly observed SNPs merely
            # because panel deletions bridge them into an oversized component.
            # In this conservative read-dominant fallback only single-base SNPs
            # with at least 20 independent ALT molecules are admissible. Keep
            # the original rejection for every unobserved/indel candidate.
            strong=[v for v in cluster if len(v[1])==len(v[2])==1 and support.get(v,0)>=20]
            selected.extend(strong);rejected.extend(v for v in cluster if v not in strong)
            continue
        if len(cluster)>max_alleles-1:
            ranked=sorted(cluster,key=lambda v:(-support.get(v,0),-panel.get(v,(None,0))[1],v))
            selected.extend(ranked[:max_alleles-1]);rejected.extend(ranked[max_alleles-1:])
        else:selected.extend(cluster)
    return sorted(selected),sorted(rejected)

def minimal_edit(variant):
    pos,ref,alt=variant
    while ref and alt and ref[0]==alt[0]:pos+=1;ref=ref[1:];alt=alt[1:]
    while ref and alt and ref[-1]==alt[-1]:ref=ref[:-1];alt=alt[:-1]
    return pos,ref,alt

def compatible_edits(edits):
    for i,(p,ref,alt) in enumerate(edits):
        for q,other_ref,other_alt in edits[:i]:
            if not ref and not other_ref:
                if p==q:return False
            elif not ref:
                if q<p<q+len(other_ref):return False
            elif not other_ref:
                if p<q<p+len(ref):return False
            elif max(p,q)<min(p+len(ref),q+len(other_ref)):return False
    return True

def apply_edits(reference,start,edits):
    result=reference
    # At an identical boundary delete/replace before inserting; all edits are
    # expressed in original reference coordinates and applied right to left.
    for pos,ref,alt in sorted(edits,key=lambda x:(x[0],len(x[1])),reverse=True):
        offset=pos-start;result=result[:offset]+alt+result[offset+len(ref):]
    return result

def make_blocks(variants,fetch,max_distance=500,panel=None,support=None,max_alleles=12,groups_override=None,fragment_seeds=None,proposal_vectors=None,catalog_cache=None):
    panel=panel or {};support=support or {}
    groups=overlap_groups(variants) if groups_override is None else groups_override;blocks=[];i=0
    while i<len(groups):
        ids=groups[i]
        if len(ids)==1 and i+1<len(groups) and len(groups[i+1])==1:
            a,b=ids[0],groups[i+1][0]
            if len(variants[a][1])==len(variants[a][2])==len(variants[b][1])==len(variants[b][2])==1 and variants[b][0]-variants[a][0]<=max_distance:
                blocks.append({'ids':[a,b],'kind':'snv_pair','vectors':np.array(list(itertools.product([0,1],repeat=2)),dtype=int)});i+=2;continue
        if len(ids)==1:
            blocks.append({'ids':ids,'kind':'single','vectors':np.array([[0],[1]],dtype=int)});i+=1;continue
        proposed=[tuple(map(int,v)) for v in (proposal_vectors or {}).get(tuple(ids),[]) if len(v)==len(ids)]
        # This cache is scoped to one immutable candidate/panel/read catalog.
        # For <=11 seeds the exhaustive candidate set is independent of the
        # later LD proposals, so only the final promotion/ranking is repeated.
        cache_key=(tuple(ids),max_alleles)
        cached=catalog_cache.get(cache_key) if catalog_cache is not None and len(ids)<=11 else None
        if cached is not None:
            start,end,reference,edits,observed,by_sequence,ranked,reserved=cached
        else:
            start=variants[ids[0]][0];end=max(variants[j][0]+len(variants[j][1]) for j in ids);reference=fetch(start,end).upper()
            edits=[minimal_edit(variants[j]) for j in ids];observed=collections.Counter();nh=len(next(iter(panel.values()))[0]) if panel else 0
            if nh:
                matrix=np.array([panel[variants[j]][0] if variants[j] in panel else np.zeros(nh) for j in ids]).T
                valid=np.all((matrix==0)|(matrix==1),axis=1)
                for vector,count in zip(*unique_rows(matrix[valid],return_counts=True)):observed[tuple(vector)]=int(count)
            patterns=collections.Counter()
            if fragment_seeds is not None:
                by_fragment=collections.defaultdict(set)
                for j,index in enumerate(ids):
                    for fragment in fragment_seeds.get(variants[index],()):by_fragment[fragment].add(j)
                for supported_ids in by_fragment.values():
                    pattern=tuple(int(j in supported_ids) for j in range(len(ids)))
                    patterns[pattern]+=1
            else:
                for j,index in enumerate(ids):
                    if support.get(variants[index],0)>=2:patterns[tuple(int(q==j) for q in range(len(ids)))]=support[variants[index]]
            patterns=collections.Counter({key:count for key,count in patterns.items() if count>=2})
            candidates=[]
            if len(ids)<=11:
                vectors_to_consider=itertools.product([0,1],repeat=len(ids))
            else:
                # Complete observed panel encodings plus read-seeded singleton/pairs.
                # Avoid an exponential enumeration when context joins dense markers.
                vectors_to_consider=set(observed);vectors_to_consider.add((0,)*len(ids))
                for j in range(len(ids)):
                    vector=[0]*len(ids);vector[j]=1;vectors_to_consider.add(tuple(vector))
                supported=[j for j in range(len(ids)) if support.get(variants[ids[j]],0)>=2]
                for a,b in itertools.combinations(supported,2):
                    vector=[0]*len(ids);vector[a]=vector[b]=1;vectors_to_consider.add(tuple(vector))
                for background,_ in sorted(observed.items(),key=lambda item:(-item[1],item[0]))[:32]:
                    for j in supported:
                        vector=list(background);vector[j]=1;vectors_to_consider.add(tuple(vector))
                backgrounds=[vector for vector,_ in sorted(observed.items(),key=lambda item:(-item[1],item[0]))[:32]]+[(0,)*len(ids)]
                for pattern,count in sorted(patterns.items(),key=lambda item:(-item[1],item[0]))[:32]:
                    for background in backgrounds:
                        vector=tuple(int(x or y) for x,y in zip(pattern,background));vectors_to_consider.add(vector)
                vectors_to_consider.update(proposed)
                vectors_to_consider=sorted(vectors_to_consider)
            for vector in vectors_to_consider:
                active=[edits[j] for j,x in enumerate(vector) if x]
                if not compatible_edits(active):continue
                sequence=apply_edits(reference,start,active)
                evidence=min((support.get(variants[ids[j]],0) for j,x in enumerate(vector) if x),default=0)
                candidates.append((sequence,vector,observed[vector],evidence))
            # Reference is always available. Prefer observed panel sequences and
            # read-supported combinations under a fixed local sequence budget.
            by_sequence=collections.defaultdict(list)
            for sequence,vector,count,evidence in candidates:by_sequence[sequence].append((vector,count,evidence))
            ranked=sorted(by_sequence,key=lambda seq:(seq!=reference,-sum(x[1] for x in by_sequence[seq]),-max(x[2] for x in by_sequence[seq]),min(sum(x[0]) for x in by_sequence[seq]),seq))
            # Reserve half the nonreference budget for paths supported by actual
            # fragment-level ALT co-occurrence, including panel-present alleles.
            # Missing read markers are unknown, not forced reference: choose their
            # most frequent compatible panel background. No truth is consulted.
            reserved=[];quota=max(1,(max_alleles-1)//2)
            for pattern,count in sorted(patterns.items(),key=lambda item:(-item[1],item[0])):
                if any(any(all(not bit or vector[j] for j,bit in enumerate(pattern)) for vector,_,_ in by_sequence[seq]) for seq in reserved):continue
                forced=[j for j,bit in enumerate(pattern) if bit];hidden=[j for j,bit in enumerate(pattern) if not bit];choices=[];background_counts=collections.Counter()
                for donor,n in observed.items():
                    background=tuple(donor[j] for j in hidden)
                    background_counts[background]+=n
                for sequence,encodings in by_sequence.items():
                    if sequence==reference or sequence in reserved:continue
                    for vector,_,evidence in encodings:
                        if not all(vector[j] for j in forced):continue
                        background_count=background_counts[tuple(vector[j] for j in hidden)]
                        choices.append((-background_count,-evidence,sum(vector),sequence))
                if choices:reserved.append(min(choices)[-1])
                if len(reserved)>=quota:break
            if catalog_cache is not None and len(ids)<=11:
                catalog_cache[cache_key]=(start,end,reference,edits,observed,by_sequence,ranked,reserved)
        # v0.2: an initial LD posterior may promote two compatible external
        # sequence paths, after the complete existing read-supported quota.
        # Original full-panel counts remain unchanged for allele frequencies.
        promoted=[]
        for vector in proposed:
            active=[edits[j] for j,x in enumerate(vector) if x]
            if not compatible_edits(active):continue
            sequence=apply_edits(reference,start,active)
            if sequence in by_sequence and sequence!=reference and sequence not in reserved and sequence not in promoted:
                promoted.append(sequence)
            if len(promoted)>=2:break
        chosen=[reference]+reserved+promoted[:max(0,max_alleles-1-len(reserved))]
        chosen += [seq for seq in ranked if seq not in chosen][:max_alleles-len(chosen)]
        alleles=chosen[:1]+sorted(chosen[1:]);representations=[[x[0] for x in by_sequence[seq]] for seq in alleles]
        vectors=np.array([reps[0] for reps in representations],dtype=int)
        denominator=sum(observed.values());counts=[sum(x[1] for x in by_sequence[seq]) for seq in alleles];af=[count/denominator if denominator and count else None for count in counts]
        blocks.append({'ids':ids,'kind':'multiallelic','start':start,'end':end,'alleles':alleles,'vectors':vectors,'representations':representations,'allele_af':af,'omitted_compatible_sequences':len(by_sequence)-len(alleles),'ld_promoted_sequences':promoted})
        i+=1
    return blocks

def likelihood_matrix(lh,artifact_prior=None):
    """Marginal read likelihood for every ordered local diploid allele pair."""
    if len(lh)==0:return _empty_likelihood(lh.shape[1],artifact_prior).copy()
    if lh.shape[1]==2 and artifact_prior is not None:
        from mixture_cpu import two
        return two(lh,artifact_prior)
    from mixture_cpu import general
    return general(lh,artifact_prior)

@lru_cache(maxsize=128)
def _empty_likelihood(count,artifact_prior):
    return _likelihood_matrix(np.empty((0,count)),artifact_prior)

def _likelihood_matrix(lh,artifact_prior=None):
    count=lh.shape[1];a=np.repeat(np.arange(count),count);b=np.tile(np.arange(count),count);la,lb=lh[:,a],lh[:,b]
    if artifact_prior is None:return np.log(np.maximum(1e-300,(la+lb)/2)).sum(axis=0).reshape(count,count)
    clean=np.array([np.log(np.maximum(1e-300,w*la+(1-w)*lb)).sum(axis=0)+math.log(p) for w,p in zip(BALANCE,BALANCE_WEIGHT)])
    clean=np.logaddexp.reduce(clean,axis=0)
    if artifact_prior:
        art=[]
        for theta,weight in zip(CONTAMINATION,CONTAMINATION_WEIGHT):
            for c in range(count):art.append(np.log(np.maximum(1e-300,(1-theta)*(la+lb)/2+theta*lh[:,c,None])).sum(axis=0)+math.log(weight/count))
        art=np.logaddexp.reduce(np.array(art),axis=0);clean=np.logaddexp(math.log1p(-artifact_prior)+clean,math.log(artifact_prior)+art)
    return clean.reshape(count,count)

def _sequence_inputs(block,reads,fetch,min_bq):
    from alignment_queries import window_bounds
    start,end=block['start'],block['end'];left=block.get('left',max(0,start-80));right=block.get('right',end+81)
    prefix,suffix=fetch(left,start).upper(),fetch(end,right).upper();haps=[prefix+a+suffix for a in block['alleles']];fragments={}
    partial=block.get('partial_reads',False);spans=block.get('marker_spans',[(start,end)])
    reps=block.get('representations',[[v] for v in block.get('vectors',[[0],[1]])])
    candidates=reads.overlap(start,end) if partial and hasattr(reads,'overlap') else (reads.at(start) if hasattr(reads,'at') and not partial else reads)
    used_partial=0;pending=[]
    for library,r in candidates:
        if r.query_qualities is None:continue
        if not partial and (r.reference_start>start-3 or r.reference_end<end+3):continue
        visible=[j for j,(a,b) in enumerate(spans) if r.reference_start<=a-3 and r.reference_end>=b+3]
        if partial and not visible:continue
        local_left,local_right=left,right
        if partial and len(visible)<len(spans):
            used_partial+=1
            first,last=min(spans[j][0] for j in visible),max(spans[j][1] for j in visible)
            local_left=max([left]+[b for j,(a,b) in enumerate(spans) if j not in visible and b<=first])
            local_right=min([right]+[a for j,(a,b) in enumerate(spans) if j not in visible and a>=last])
        bounds=window_bounds(r,local_left,local_right)
        if bounds is None:continue
        q0,q1=bounds
        if q1-q0<12:continue
        seq=r.query_sequence[q0:q1].upper();q=bytes(r.query_qualities[q0:q1])
        if not q or sum(q)/len(q)<min_bq:continue
        pending.append((library,r,visible,q0,q1,seq,q))
    return dict(haps=haps,pending=pending,partial=partial,spans=spans,reps=reps,used_partial=used_partial)

def _finish_sequence(context,likelihoods):
    haps=context['haps'];pending=context['pending'];partial=context['partial'];spans=context['spans'];reps=context['reps'];used_partial=context['used_partial'];fragments={}
    for (library,r,visible,q0,q1,seq,q),ll in zip(pending,likelihoods):
        values=np.exp(ll-ll.max())
        if partial and len(visible)<len(spans):
            # A partial alignment cannot distinguish alleles that differ only at
            # unobserved markers. Marginalize retained hidden sequence contexts
            # uniformly, making their likelihood exactly equal. This bounded
            # nuisance-context integration is approximate, not a global read HMM.
            classes=collections.defaultdict(list)
            for allele,encodings in enumerate(reps):
                key=tuple(sorted({tuple(vector[j] for j in visible) for vector in encodings}))
                classes[key].append(allele)
            for alleles in classes.values():values[alleles]=values[alleles].mean()
        me=10**(-min(r.mapping_quality,60)/10);values=(1-me)*values+me*values.mean()
        certainty=float(np.ptp(np.log(np.maximum(1e-300,values))));key=(library,r.query_name)
        # v0.2 instrumentation: preserve the exact accepted molecular evidence
        # and its quality metadata without changing any likelihood calculation.
        visible_fraction=len(visible)/max(1,len(spans)) if partial else 1.
        fragments.setdefault(key,[]).append((certainty,r.reference_start,r.reference_end,values,(r,q0,q1,visible_fraction)))
    combined=[];fragment_keys=[];fragment_alignments=[]
    for key,alignments in fragments.items():
        accepted=[]
        for item in sorted(alignments,key=lambda x:-x[0]):
            if any(max(item[1],other[1])<min(item[2],other[2]) for other in accepted):continue
            accepted.append(item)
        logs=sum((np.log(np.maximum(1e-300,item[3])) for item in accepted),np.zeros(len(haps)))
        combined.append(np.exp(logs-logs.max()))
        fragment_keys.append(key);fragment_alignments.append([item[4] for item in accepted])
    lh=np.vstack(combined) if combined else np.empty((0,len(haps)))
    ad=np.zeros(len(haps),dtype=int)
    for row in lh:
        ranked=np.argsort(-row);best,second=ranked[:2]
        if row[best]>=3*row[second]:ad[best]+=1
    return {'lh':lh,'dp':len(lh),'ad':ad,'partial_alignments':used_partial,
            'fragment_keys':fragment_keys,'fragment_alignments':fragment_alignments}

def single_base_likelihoods(block,reads,fetch,min_bq):
    """Direct categorical base evidence needs no artificial alignment window.

    A nearby variant can clip a sequence window below 12 bases. Multiallelic
    SNPs still have a valid observed base; discarding all their reads is wrong.
    Overlapping mates remain one molecular observation, as for other sites.
    """
    from alignment_queries import target_queries
    pos=block['start'];alleles=block['alleles'];n=len(alleles)
    if block['end']!=pos+1 or any(len(a)!=1 or a not in 'ACGT' for a in alleles):raise ValueError('Single canonical base alleles required')
    fragments={}
    candidates=reads.at(pos) if hasattr(reads,'at') else reads
    for library,read in candidates:
        if read.query_qualities is None:continue
        for q,p in target_queries(read,[pos]):
            if p!=pos or read.query_qualities[q]<min_bq:continue
            base=read.query_sequence[q].upper()
            if base not in 'ACGT':continue
            error=10**(-min(int(read.query_qualities[q]),60)/10);mapping=10**(-min(read.mapping_quality,60)/10)
            values=np.array([(1-mapping)*(1-error if a==base else error/3)+mapping/4 for a in alleles])
            certainty=float(np.log(values.max()/values.min()));key=(library,read.query_name)
            if key not in fragments or certainty>fragments[key][0]:fragments[key]=(certainty,values/values.max(),read,q)
    keys=list(fragments);lh=np.array([fragments[k][1] for k in keys]) if keys else np.empty((0,n))
    ad=np.zeros(n,dtype=int)
    for values in lh:
        ranked=np.argsort(-values);a,b=ranked[:2]
        if values[a]>=3*values[b]:ad[a]+=1
    return dict(lh=lh,dp=len(keys),ad=ad,partial_alignments=0,fragment_keys=keys,
                fragment_alignments=[[(fragments[k][2],fragments[k][3],fragments[k][3]+1,1.)] for k in keys])

def sequence_likelihoods(block,reads,fetch,min_bq,lib):
    from caller import pair_scores_many
    context=_sequence_inputs(block,reads,fetch,min_bq);pending=context['pending']
    # Retain sequence-aware evidence where it is available. A clipped-context
    # rescue requires >=20 distinct molecules, matching the existing read-
    # dominant model regime; low-depth imputation emissions remain frozen.
    if not pending and block['end']==block['start']+1 and all(len(a)==1 for a in block['alleles']):
        rescued=single_base_likelihoods(block,reads,fetch,min_bq)
        if rescued['dp']>=20:return rescued
    likelihoods=pair_scores_many(lib,[x[5] for x in pending],[x[6] for x in pending],context['haps'])
    return _finish_sequence(context,likelihoods)

def sequence_likelihoods_many(blocks,reads,fetch,min_bq,lib,max_pairs=32768):
    import os
    if os.environ.get('VARDEPTH_PAIRHMM_BACKEND','cpu')=='cpu':
        return [sequence_likelihoods(block,reads,fetch,min_bq,lib) for block in blocks]
    from accelerator import score_pairs
    contexts=[];jobs=[];result=[]
    def flush():
        if not contexts:return
        likelihoods=score_pairs(jobs);offset=0
        for context in contexts:
            shape=(len(context['pending']),len(context['haps']));n=shape[0]*shape[1]
            result.append(_finish_sequence(context,likelihoods[offset:offset+n].reshape(shape)));offset+=n
        contexts.clear();jobs.clear()
    for block in blocks:
        context=_sequence_inputs(block,reads,fetch,min_bq)
        n=len(context['pending'])*len(context['haps'])
        if jobs and len(jobs)+n>max_pairs:flush()
        contexts.append(context)
        jobs.extend((x[5],x[6],h) for x in context['pending'] for h in context['haps'])
        if len(jobs)>=max_pairs:flush()
    flush();return result

def ordinary_likelihoods(block,variants,observations):
    from artifact import ordinary_rows
    return ordinary_rows(variants,block['ids'],block['vectors'].tolist(),observations)


def local_infer(lib,variants,hap,gl,mu,observations,blocks,sequence_data,no_ld=False,artifact_prior=None,recombination=None,indel_artifact_prior=None,factor_cache=None):
    """Exact diploid block HMM conditional on bounded compatible local alleles."""
    m,k=hap.shape;kk=k*k;emissions=np.zeros((len(blocks),kk));eg=np.zeros((m,kk,3));pn=np.zeros((m,kk,2));bof=np.empty(m,dtype=np.int32);local=[];logscale=0
    allele_one=mu[:,None]+(1-2*mu[:,None])*hap;allele_zero=1-allele_one
    shared_factors=factor_cache.setdefault('_by_content',OrderedDict()) if factor_cache is not None else OrderedDict()
    if factor_cache is not None and factor_cache.get('_observation_source') is not observations:
        factor_cache['_observation_source']=observations
        factor_cache['_ordinary_lh']={}
    for t,block in enumerate(blocks):
        ids=block['ids'];vectors=block['vectors'];representations=block.get('representations',[[vector] for vector in vectors])
        if block['kind']=='multiallelic':lh=sequence_data[t]['lh']
        else:
            ordinary_cache=factor_cache.setdefault('_ordinary_lh',{}) if factor_cache is not None else {}
            lh_key=(tuple(ids),vectors.shape,vectors.tobytes())
            if lh_key not in ordinary_cache:ordinary_cache[lh_key]=ordinary_likelihoods(block,variants,observations)
            lh=ordinary_cache[lh_key]
        prior=prior_for(variants,ids,artifact_prior,indel_artifact_prior)
        key=(block['kind'],len(ids),tuple(tuple(tuple(map(int,v)) for v in reps) for reps in representations),hap[ids].tobytes(),mu[ids].tobytes(),prior,lh.shape,lh.tobytes())
        cached=factor_cache.get(t) if factor_cache is not None else None
        if cached is None or cached[0]!=key:cached=shared_factors.get(key)
        if cached is not None and cached[0]==key:
            if key in shared_factors:shared_factors.move_to_end(key)
            _,e,g,phase_n,item,scale=cached
            if factor_cache is not None:factor_cache[t]=cached
            emissions[t]=e;eg[ids]=g;pn[ids]=phase_n;bof[ids]=t;local.append(item);logscale+=scale
            continue
        prob=np.zeros((len(vectors),k));marker_one=np.zeros((len(ids),len(vectors),k))
        for local_allele,reps in enumerate(representations):
            for vector in reps:
                weight=np.ones(k)
                for local_marker,j in enumerate(ids):
                    bof[j]=t;weight*=allele_one[j] if vector[local_marker] else allele_zero[j]
                prob[local_allele]+=weight
                for local_marker in range(len(ids)):
                    bit=vector[local_marker]
                    if bit==1:marker_one[local_marker,local_allele]+=weight
                    elif bit!=0:marker_one[local_marker,local_allele]+=weight*bit
        # Restrict and renormalize the donor allele distribution to physically
        # compatible alternatives. No two competing indels coexist on one copy.
        denominator=prob.sum(axis=0);prob/=denominator[None,:];marker_one/=denominator[None,None,:]
        log_likelihood=likelihood_matrix(lh,prior);scale=float(log_likelihood.max());logscale+=scale;L=np.exp(log_likelihood-scale)
        prob_l=prob.T@L;E=prob_l@prob;emissions[t]=E.reshape(-1)
        for local_marker,j in enumerate(ids):
            one=marker_one[local_marker];zero=prob-one;one_l=one.T@L;zero_l=zero.T@L
            eg[j,:,0]=(zero_l@zero).reshape(-1);eg[j,:,1]=(one_l@zero+zero_l@one).reshape(-1);eg[j,:,2]=(one_l@one).reshape(-1)
            pn[j,:,0]=(one_l@prob).reshape(-1);pn[j,:,1]=(prob_l@one).reshape(-1)
        local.append((prob,L,log_likelihood))
        entry=(key,emissions[t].copy(),eg[ids],pn[ids],local[-1],scale)
        if factor_cache is not None:factor_cache[t]=entry
        shared_factors[key]=entry
        if len(shared_factors)>4096:shared_factors.popitem(last=False)
    pos=np.array([variants[b['ids'][0]][0] for b in blocks]);dist=np.diff(pos,prepend=pos[0]);r=np.ones(len(blocks)) if no_ld else np.clip(1-np.exp(-dist*1e-6),1e-8,.5)
    if recombination:
        from genetic_map import transitions
        r=transitions(pos,no_ld=no_ld,**recombination)
    gp=np.empty_like(gl);sp=np.empty_like(emissions);p=lambda x:x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    lib.block_hmm.argtypes=[ctypes.c_int]*3+[ctypes.POINTER(ctypes.c_double)]*2+[ctypes.POINTER(ctypes.c_int)]+[ctypes.POINTER(ctypes.c_double)]*3;lib.block_hmm.restype=ctypes.c_double
    ll=lib.block_hmm(len(blocks),m,k,p(emissions),p(eg),bof.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),p(r),p(gp),p(sp))
    phase=np.zeros((m,2));joint={};ld_proposals={}
    best_state=np.argmax(sp,axis=1)[bof]
    phase=pn[np.arange(m),best_state]/np.maximum(1e-300,emissions[bof,best_state])[:,None]
    for t,block in enumerate(blocks):
        if block['kind']!='multiallelic':continue
        # Posterior donor occupancy is measured after forward/backward over LD,
        # not only the initial marginal state-selection score. This is used
        # exclusively for candidate proposals, never as a likelihood/prior.
        state=sp[t].reshape(k,k);usage=state.sum(axis=0)+state.sum(axis=1)
        weights=collections.Counter()
        local_hap=hap[block['ids']]
        valid_donors=np.all((local_hap==0)|(local_hap==1),axis=0)&np.any(local_hap,axis=0)
        for donor in np.flatnonzero(valid_donors):
            vector=local_hap[:,donor]
            weights[tuple(map(int,vector))]+=float(usage[donor])
        ld_proposals[block['ids'][0]]=[{'vector':list(v),'weight':w} for v,w in sorted(weights.items(),key=lambda x:(-x[1],x[0]))[:2]]
        prob,L,logL=local[t];state_factor=(sp[t]/np.maximum(1e-300,emissions[t])).reshape(k,k)
        posterior=L*(prob@state_factor@prob.T);posterior/=posterior.sum();genotypes=[(a,b) for b in range(len(prob)) for a in range(b+1)]
        GP=np.array([posterior[a,b]+(posterior[b,a] if a!=b else 0) for a,b in genotypes]);PL=np.array([logL[a,b] for a,b in genotypes]);PL-=PL.max()
        joint[block['ids'][0]]={'ids':block['ids'],'start':block['start'],'alleles':block['alleles'],'allele_af':block.get('allele_af'),'genotypes':genotypes,'gp':GP,'pl':PL,'dp':sequence_data[t]['dp'],'ad':sequence_data[t]['ad']}
    details={'log_marginal_likelihood':ll+logscale,'blocks':len(blocks),'multiallelic_blocks':len(joint),'joint_records':joint,'ld_sequence_proposals':ld_proposals}
    import os
    if os.environ.get('VARDEPTH_TRACE_EVIDENCE','0')=='1':
        from posterior_summary import summarize
        details['hmm_uncertainty']=summarize(sp,k)[bof]
    return gp,phase,details
