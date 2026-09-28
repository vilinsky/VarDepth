"""Exact lexicographic row deduplication with packed binary haplotype keys."""
import numpy as np

def unique_rows(matrix,*,return_counts=False,return_inverse=False):
    x=np.asarray(matrix)
    if x.ndim!=2:raise ValueError('Expected matrix')
    if x.shape[1]==1:
        key=x[:,0]
    elif x.shape[1]<=64 and np.all((x==0)|(x==1)):
        packed=np.packbits(x.astype(np.uint8,copy=False),axis=1,bitorder='big')
        # Big-endian packed bytes have exactly the original lexicographic order.
        key=packed[:,0] if packed.shape[1]==1 else packed.view(np.dtype((np.void,packed.shape[1]))).reshape(-1)
    else:return np.unique(x,axis=0,return_counts=return_counts,return_inverse=return_inverse)
    found=np.unique(key,return_index=True,return_inverse=return_inverse,return_counts=return_counts)
    result=[x[found[1]]]
    result.extend(found[2:])
    return tuple(result) if return_inverse or return_counts else result[0]
