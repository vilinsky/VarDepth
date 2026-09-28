"""Batch independent per-alignment arithmetic; retain each row's reduction."""
import numpy as np

_MQ_ERROR=np.array([10**(-min(mq,60)/10) for mq in range(256)],np.float64)

def normalized_rows(likelihoods,mapping_qualities):
    ll=np.asarray(likelihoods,dtype=np.float64)
    if ll.ndim!=2 or len(ll)!=len(mapping_qualities):raise ValueError('Likelihood rows and MQ differ')
    if not len(ll):return ll.copy(),np.empty(0,np.float64)
    values=np.exp(ll-ll.max(axis=1,keepdims=True))
    error=_MQ_ERROR[np.asarray(mapping_qualities,np.uint8),None]
    values=(1-error)*values+error*values.mean(axis=1,keepdims=True)
    certainty=np.ptp(np.log(np.maximum(1e-300,values)),axis=1)
    return values,certainty
