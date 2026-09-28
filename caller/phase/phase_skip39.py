"""Panel HMM bridges over donor-uninformative heterozygotes; no truth input."""
import numpy as np
from hybrid_phase import transition

def phase_edges_bridged(haplotypes, genotypes, refresh, mutation=.001):
    """Exact adjacent-het and informative-flank parity under a donor-pair HMM.

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
    # A heterozygote with no donor carrying one of its alleles has no copying
    # orientation. Keep its emission, but do not let it hide the phase
    # relationship between informative flanking heterozygotes.
    informative=het[(h[het].min(axis=1)<.5)&(h[het].max(axis=1)>.5)]
    pairs=set(zip(het[:-1],het[1:]))|set(zip(informative[:-1],informative[1:]))
    for left,right in sorted(pairs):
        a=fw[left]*orientation(left)
        for t in range(left+1,right+1):a=transition(a,r[t])*emission(t)/scale[t]
        denominator=(fw[right]*back[right]).sum()
        # Exchange symmetry gives P(01)+P(10)=2 P(01).
        p=float(2*(a*back[right]*(1-orientation(right))).sum()/denominator)
        if not -1e-8<=p<=1+1e-8:raise ArithmeticError('Invalid edge posterior')
        result.append((int(left),int(right),min(1.,max(0.,p))))
    return result

