"""Experimental classical phasing: exact donor-pair HMM and cautious link graph.

No truth, parental labels, or genotype changes enter this module. Edge errors
are model quantities, not a claim of empirical clinical calibration.
"""
from dataclasses import dataclass
import math
import numpy as np


def transition(a, refresh):
    """T A T' for independent uniform donor refresh, in O(K^2)."""
    k=len(a);r=float(refresh);s=1-r
    return s*s*a+s*r/k*(a.sum(0)[None,:]+a.sum(1)[:,None])+(r/k)**2*a.sum()


def phase_edges(haplotypes, genotypes, refresh, mutation=.001):
    """Exact adjacent-het trans probabilities conditional on a donor subset.

    Genotypes are fixed 0/1/2 dosages, not multiplied read/panel posterior
    experts. Allelic mutation is integrated, including heterozygotes copied
    from locally equal donor alleles. No arbitrary haplotype-label anchoring.
    """
    h=np.asarray(haplotypes,float);g=np.asarray(genotypes,int);r=np.asarray(refresh,float)
    if h.ndim!=2 or g.shape!=(len(h),) or r.shape!=g.shape or h.shape[1]<2:
        raise ValueError('Invalid diploid phase dimensions')
    if (not np.isfinite(h).all() or not np.isfinite(r).all() or
        np.any((h<0)|(h>1)) or np.any((r<0)|(r>1)) or
        np.any((g<0)|(g>2)) or not 0<mutation<.5):raise ValueError('Invalid phase inputs')
    m,k=h.shape
    if not m:return []
    x=mutation+(1-2*mutation)*h
    def emission(t):
        a=x[t,:,None];b=x[t,None,:]
        if g[t]==0:return (1-a)*(1-b)
        if g[t]==2:return a*b
        return a*(1-b)+(1-a)*b
    def orientation(t):
        a=x[t,:,None];b=x[t,None,:]
        return (1-a)*b/emission(t)  # first true chromosome carries REF
    fw=np.empty((m,k,k));scale=np.empty(m)
    previous=np.full((k,k),1/(k*k))
    for t in range(m):
        current=transition(previous,r[t])*emission(t);scale[t]=current.sum()
        if not np.isfinite(scale[t]) or scale[t]<=0:raise ArithmeticError('Invalid HMM normalization')
        fw[t]=current/scale[t];previous=fw[t]
    # Only store backward messages at heterozygotes; forward is needed for all.
    back={};b=np.ones((k,k))
    for t in range(m-1,-1,-1):
        if g[t]==1:back[t]=b.copy()
        if t:b=transition(emission(t)*b,r[t])/scale[t]
    het=np.flatnonzero(g==1);result=[]
    for left,right in zip(het[:-1],het[1:]):
        a=fw[left]*orientation(left)
        for t in range(left+1,right+1):a=transition(a,r[t])*emission(t)/scale[t]
        denominator=(fw[right]*back[right]).sum()
        # Exchange symmetry gives P(01)+P(10)=2 P(01).
        p=float(2*(a*back[right]*(1-orientation(right))).sum()/denominator)
        if not -1e-8<=p<=1+1e-8:raise ArithmeticError('Invalid edge posterior')
        result.append((int(left),int(right),min(1.,max(0.,p))))
    return result


@dataclass(frozen=True)
class Edge:
    left:int
    right:int
    parity:int
    error:float
    source:str
    support:int=0


def phase_forest(count, edges, max_error=.01):
    """Confidence forest, local conflict quarantine, bounded pair-path error.

    A block's maximum weighted path (tree diameter) bounds any queried pair
    by the union bound. Unrelated branches do not consume that pair's budget.
    Contradictory cycles quarantine every site on the inconsistent path, so
    unaffected tree branches survive as separate blocks, never joined across
    the conflict. Edge calibration remains an empirical validation requirement.
    """
    if not 0<max_error<.5:raise ValueError('Invalid block error budget')
    parent=list(range(count));xor=[0]*count;size=[1]*count
    def root(i):
        value=0
        while parent[i]!=i:value^=xor[i];i=parent[i]
        return i,value
    accepted=[];contradictions=[];rejected=0;adj=[[] for _ in range(count)];bad=set()
    def path_nodes(a,b):
        previous={a:None};todo=[a]
        while todo:
            v=todo.pop()
            if v==b:break
            for w,e in adj[v]:
                if w not in previous:previous[w]=v;todo.append(w)
        if b not in previous:raise AssertionError('Connected forest path missing')
        result=[];v=b
        while v is not None:result.append(v);v=previous[v]
        return result
    ordered=sorted(edges,key=lambda e:(e.error,e.left,e.right,e.source))
    for edge in ordered:
        if not(0<=edge.left<count and 0<=edge.right<count and edge.left!=edge.right and
               edge.parity in (0,1) and math.isfinite(edge.error) and 0<=edge.error<=.5):
            raise ValueError('Invalid phase edge')
        if edge.error>max_error:rejected+=1;continue
        a,xa=root(edge.left);b,xb=root(edge.right)
        if a==b:
            if xa^xb!=edge.parity:
                contradictions.append((edge.left,edge.right));bad.update(path_nodes(edge.left,edge.right))
            continue
        if size[a]<size[b]:a,b=b,a
        parent[b]=a;xor[b]=xa^xb^edge.parity
        size[a]+=size[b];accepted.append(edge)
        adj[edge.left].append((edge.right,edge.error));adj[edge.right].append((edge.left,edge.error))
    part=list(range(count));psize=[1]*count;diameter=[0.]*count;kept=[[] for _ in range(count)];nkept=0
    def component(i):
        while part[i]!=i:i=part[i]
        return i
    def extent(start):
        maximum=0.;todo=[(start,-1,0.)]
        while todo:
            v,previous,d=todo.pop();maximum=max(maximum,d)
            todo.extend((w,v,d+e) for w,e in kept[v] if w!=previous)
        return maximum
    for edge in accepted:
        if edge.left in bad or edge.right in bad:continue
        a=component(edge.left);b=component(edge.right)
        proposed=max(diameter[a],diameter[b],extent(edge.left)+edge.error+extent(edge.right))
        if proposed>max_error:rejected+=1;continue
        if psize[a]<psize[b]:a,b=b,a
        part[b]=a;psize[a]+=psize[b];diameter[a]=proposed;nkept+=1
        kept[edge.left].append((edge.right,edge.error));kept[edge.right].append((edge.left,edge.error))
    members={}
    for i in range(count):
        a=component(i);_,x=root(i)
        if psize[a]>1 and i not in bad:members.setdefault(a,[]).append((i,x))
    output={}
    for a,group in members.items():
        first,origin=min(group)
        for i,x in group:output[i]=(first,x^origin,diameter[a])
    return output,dict(accepted_edges=nkept,budget_rejected_edges=rejected,
        contradictory_cycles=len(contradictions),withheld_conflict_sites=len(bad),
        withheld_components=len({root(i)[0] for i in bad}),blocks=len(members),error_budget='maximum pair-path union bound')


def allele_from_read(read, start, alleles, min_baseq=20):
    """Conservative literal allele observation with two aligned flanks.

    Equivalent repeat-shifted CIGARs are deliberately missing observations;
    they are never silently called REF. Local realignment is a later extension.
    """
    ref=alleles[0];end=start+len(ref)
    if read.reference_start>start-1 or read.reference_end is None or read.reference_end<=end:return None
    seq=read.query_sequence;qual=read.query_qualities
    if seq is None or qual is None:return None
    p=read.reference_start;q=0;text=[];qualities=[];left=False;right=False;nearby_indels=[]
    for op,n in read.cigartuples or []:
        if op in (0,7,8):
            if p<=start-1<p+n:left=True;qualities.append(int(qual[q+start-1-p]))
            if p<=end<p+n:right=True;qualities.append(int(qual[q+end-p]))
            lo=max(p,start);hi=min(p+n,end)
            if hi>lo:
                a=q+lo-p;b=q+hi-p;text.append(seq[a:b]);qualities.extend(qual[a:b])
            p+=n;q+=n
        elif op==1:
            nearby_indels.append((p,op,n))
            if start<p<=end:text.append(seq[q:q+n]);qualities.extend(qual[q:q+n])
            q+=n
        elif op==2:
            nearby_indels.append((p,op,n))
            p+=n
        elif op==3:
            if p<end+1 and p+n>start-1:return None
            p+=n
        elif op==4:q+=n
        elif op in (5,6):pass
        else:return None
    if not left or not right or not qualities or min(qualities)<min_baseq:return None
    observed=''.join(text).upper();matches=[i for i,a in enumerate(alleles) if a.upper()==observed]
    if len(matches)!=1:return None
    if len(set(map(len,alleles)))>1 and matches[0]==0 and nearby_indels:
        # A repeat-shifted ALT CIGAR may look like REF at the canonical anchor.
        # Without realignment, abstain instead of inventing REF support.
        return None
    # Systematic alignment error floor, more conservative for length variants.
    floor=.02 if len(set(map(len,alleles)))>1 else .005
    e=max(floor,10**(-min(min(qualities),read.mapping_quality)/10))
    return matches[0],e


def molecular_edges(molecules, minimum_support=3):
    """Collapse mates first; one likelihood contribution per molecule/pair."""
    counts={}
    for observations in molecules:
        usable=sorted((i,v) for i,v in observations.items() if v is not None)
        # A spanning path contains enough constraints for each molecule.
        for (a,(aa,ea)),(b,(ab,eb)) in zip(usable[:-1],usable[1:]):
            err=ea+eb-2*ea*eb;parity=aa^ab
            score=(1 if parity==0 else -1)*math.log((1-err)/err)
            values=counts.setdefault((a,b),[0.,0]);values[0]+=score;values[1]+=1
    out=[]
    for (a,b),(score,n) in counts.items():
        if n>=minimum_support:
            # Avoid unlimited confidence from correlated alignment artifacts.
            error=max(.0001,1/(1+math.exp(min(700,abs(score)))))
            out.append(Edge(a,b,int(score<0),error,'reads',n))
    return out
