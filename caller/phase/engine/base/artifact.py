"""Exact finite-mixture local read model for research error calibration.

A shared latent third local haplotype can explain correlated alignment artifacts.
An independent clean-state allele-balance mixture protects capture-biased hets.
These fixed generic priors are development hypotheses, not measured biology.
No truth coordinates or variant-specific fitted lookups enter the model.
"""
import itertools,math
from functools import lru_cache
import numpy as np
from blocks import blocks_for

BALANCE=np.array([.2,.3,.4,.5,.6,.7,.8])
BALANCE_WEIGHT=np.array([.025,.1,.225,.3,.225,.1,.025])
CONTAMINATION=np.array([.02,.1,.2,.3])
CONTAMINATION_WEIGHT=np.array([.4,.35,.2,.05])

@lru_cache(maxsize=20000)
def observation_probabilities(bq,mq,ref_match,alt_match):
    e=10**(-bq/10);em=10**(-mq/10)
    return ((1-em)*(1-e if ref_match else e/3)+em/4,
            (1-em)*(1-e if alt_match else e/3)+em/4)

def ordinary_rows(variants,indices,vectors,observations):
    """Construct once per fragment, preserving marker multiplication order."""
    # With one marker there is no cross-marker join and multiplication by
    # the initial 1.0 leaves each FP64 likelihood unchanged.
    if len(indices)==1 and len(vectors)==2 and tuple(vectors[0])==(0,) and tuple(vectors[1])==(1,):
        j=indices[0];ref,alt=variants[j][1:]
        values=observations.get(j,{})
        if len(ref)==len(alt)==1:
            return np.asarray([observation_probabilities(min(bq,60),min(mq,60),base==ref,base==alt)
                for base,bq,mq in values.values()],dtype=np.float64).reshape(len(values),2)
        return np.asarray([obs[:2] for obs in values.values()],dtype=np.float64).reshape(len(values),2)
    fragments={}
    for local,j in enumerate(indices):
        ref,alt=variants[j][1:];snp=len(ref)==len(alt)==1
        for key,obs in observations.get(j,{}).items():
            if snp:
                base,bq,mq=obs
                p=observation_probabilities(min(bq,60),min(mq,60),base==ref,base==alt)
            else:p=obs[:2]
            fragments.setdefault(key,{})[local]=p
    result=[]
    # Python floats use the same FP64 products; avoid tiny NumPy allocations
    # for each read-marker pair. No fragments are merged or discarded.
    for calls in fragments.values():
        row=[1.]*len(vectors)
        for local,p in calls.items():
            for a,v in enumerate(vectors):row[a]*=p[v[local]]
        result.append(row)
    return np.asarray(result,dtype=np.float64).reshape(len(result),len(vectors))

def prior_for(variants,indices,artifact_prior,indel_artifact_prior=None):
    if artifact_prior is None:return None
    if indel_artifact_prior is not None and any(len(variants[j][1])!=len(variants[j][2]) for j in indices):return indel_artifact_prior
    return artifact_prior

def configurations(variants,indices,observations,artifact_prior=.01):
    localhaps=list(itertools.product([0,1],repeat=len(indices)));n=len(localhaps)
    lh=ordinary_rows(variants,indices,localhaps,observations)
    if n==2 and artifact_prior is not None:
        from mixture_cpu import two
        values=two(lh,artifact_prior).ravel()
        return [(localhaps[a],localhaps[b],float(values[2*a+b])) for a in range(2) for b in range(2)]
    from mixture_cpu import general
    values=general(lh,artifact_prior)
    return [(localhaps[a],localhaps[b],float(values[a,b])) for a in range(n) for b in range(n)]

def _configurations_numpy_lh(lh,localhaps,artifact_prior):
    n=len(localhaps)
    ai=np.repeat(np.arange(n),n);bi=np.tile(np.arange(n),n)
    la,lb=lh[:,ai],lh[:,bi]
    clean=np.array([np.log(np.maximum(1e-300,w*la+(1-w)*lb)).sum(axis=0)+math.log(p) for w,p in zip(BALANCE,BALANCE_WEIGHT)])
    clean=np.logaddexp.reduce(clean,axis=0)
    if artifact_prior:
        art=[]
        for theta,weight in zip(CONTAMINATION,CONTAMINATION_WEIGHT):
            for c in range(n):
                mixture=(1-theta)*(la+lb)/2+theta*lh[:,c,None]
                art.append(np.log(np.maximum(1e-300,mixture)).sum(axis=0)+math.log(weight/n))
        art=np.logaddexp.reduce(np.array(art),axis=0)
        ll=np.logaddexp(math.log1p(-artifact_prior)+clean,math.log(artifact_prior)+art)
    else:ll=clean
    return [(localhaps[a],localhaps[b],float(x)) for a,b,x in zip(ai,bi,ll)]

def site_likelihoods(variants,observations,artifact_prior=.01,indel_artifact_prior=None):
    gl=np.zeros((len(variants),3))
    for j in range(len(variants)):
        if not observations.get(j):continue
        config=configurations(variants,[j],observations,prior_for(variants,[j],artifact_prior,indel_artifact_prior))
        gl[j]=[config[0][2],config[1][2],config[3][2]]
    gl-=gl.max(axis=1)[:,None]
    return gl

def factors(variants,hap,gl,mu,observations,max_distance=500,artifact_prior=.01,indel_artifact_prior=None):
    blocks=blocks_for(variants,max_distance);m,k=hap.shape;kk=k*k
    emissions=np.zeros((len(blocks),kk));eg=np.zeros((m,kk,3));phase_n=np.zeros((m,kk,2));block_of=np.empty(m,dtype=np.int32);logscale=0
    for t,indices in enumerate(blocks):
        for j in indices:block_of[j]=t
        configs=configurations(variants,indices,observations,prior_for(variants,indices,artifact_prior,indel_artifact_prior))
        scale=max(ll for _,_,ll in configs);logscale+=scale
        for a,b,ll in configs:
            prior=np.ones((k,k))
            for local,j in enumerate(indices):
                x=mu[j]+(1-2*mu[j])*hap[j];pa=x if a[local] else 1-x;pb=x if b[local] else 1-x
                prior*=pa[:,None]*pb[None,:]
            w=prior.reshape(-1)*math.exp(ll-scale);emissions[t]+=w
            for local,j in enumerate(indices):
                eg[j,:,a[local]+b[local]]+=w;phase_n[j,:,0]+=a[local]*w;phase_n[j,:,1]+=b[local]*w
    return blocks,emissions,eg,phase_n,block_of,logscale
