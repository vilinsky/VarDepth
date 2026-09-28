"""VCF-order unordered diploid likelihoods, without unused ordered pairs."""
import math
import numpy as np
from artifact import BALANCE,BALANCE_WEIGHT,CONTAMINATION,CONTAMINATION_WEIGHT

def diploid_likelihood(lh,artifact_prior=None):
    if lh.shape[1]==2 and artifact_prior is not None:
        from mixture_cpu import two
        return two(lh,artifact_prior).ravel()[[0,1,3]]
    from mixture_cpu import general
    return general(lh,artifact_prior,triangular=True)

def _diploid_likelihood_numpy(lh,artifact_prior=None):
    count=lh.shape[1];pairs=[(a,b) for b in range(count) for a in range(b+1)]
    a,b=np.array(pairs,dtype=np.int64).T;la,lb=lh[:,a],lh[:,b]
    if artifact_prior is None:return np.log(np.maximum(1e-300,(la+lb)/2)).sum(axis=0)
    clean=np.array([np.log(np.maximum(1e-300,w*la+(1-w)*lb)).sum(axis=0)+math.log(p) for w,p in zip(BALANCE,BALANCE_WEIGHT)])
    clean=np.logaddexp.reduce(clean,axis=0)
    if artifact_prior:
        art=[]
        for theta,weight in zip(CONTAMINATION,CONTAMINATION_WEIGHT):
            for c in range(count):art.append(np.log(np.maximum(1e-300,(1-theta)*(la+lb)/2+theta*lh[:,c,None])).sum(axis=0)+math.log(weight/count))
        art=np.logaddexp.reduce(np.array(art),axis=0);clean=np.logaddexp(math.log1p(-artifact_prior)+clean,math.log(artifact_prior)+art)
    return clean
