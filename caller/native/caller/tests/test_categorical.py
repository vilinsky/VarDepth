import itertools
import unittest
import numpy as np
from categorical import impute


def exact(h, gl, r, mutation):
    """Independent small-state ordered diploid forward/backward oracle."""
    m,k=h.shape; states=list(itertools.product(range(k),repeat=2)); emissions=[]; local=[]
    for t in range(m):
        n=int((np.sqrt(8*len(gl[t])+1)-1)/2); pairs=[(a,b) for b in range(n) for a in range(b+1)]
        q=np.full((k,n),mutation/n)
        for donor,value in enumerate(h[t]):
            if value==255:q[donor]=1/n
            else:q[donor,value]+=1-mutation
        e=np.zeros((k*k,len(pairs))); likelihood=np.exp(gl[t]-np.max(gl[t]))
        for s,(u,v) in enumerate(states):
            for j,(a,b) in enumerate(pairs):
                e[s,j]=(q[u,a]*q[v,b]+(q[u,b]*q[v,a] if a!=b else 0))*likelihood[j]
        local.append(e);emissions.append(e.sum(1))
    f=[emissions[0]/(k*k)]; f[0]/=f[0].sum(); trans=[]
    for t in range(1,m):
        single=(1-r[t])*np.eye(k)+r[t]/k;transition=np.kron(single,single);trans.append(transition)
        x=(f[-1]@transition)*emissions[t];f.append(x/x.sum())
    b=np.ones(k*k);out=[None]*m
    for t in range(m-1,-1,-1):
        state=f[t]*b;state/=state.sum();out[t]=(state[:,None]*local[t]/emissions[t][:,None]).sum(0)
        if t:b=trans[t-1]@(emissions[t]*b);b/=b.max()
    return out


class CategoricalContracts(unittest.TestCase):
    def test_multiallelic_posterior_against_exact_diploid_hmm(self):
        h=np.array([[0,1,2],[0,1,1],[1,0,2]],np.uint8)
        gl=[np.log([.1,.8,.3,.2,.7,.1]),np.log([.9,.2,.1]),np.log([.1,.6,.2,.8,.3,.1])]
        r=np.array([0.,.14,.27]);mu=.03
        wanted=exact(h,gl,r,mu)
        observed=impute(h,gl,r,mutation=mu,burn=500,iterations=20000,chains=4,seed=617)
        for p,q in zip(observed,wanted):
            np.testing.assert_allclose(p,q,atol=.009,rtol=0)
            np.testing.assert_allclose(p.sum(),1,atol=1e-10)

    def test_missing_donor_is_marginalized(self):
        h=np.array([[0,255,2]],np.uint8); gl=[np.log([.1,.6,.4,.8,.2,.3])];r=np.array([0.])
        wanted=exact(h,gl,r,.01)
        observed=impute(h,gl,r,mutation=.01,burn=200,iterations=10000,chains=4,seed=51)
        np.testing.assert_allclose(observed[0],wanted[0],atol=.01,rtol=0)

    def test_wrong_dimensions_and_missing_encoding_rejected(self):
        for h,gl in [(np.array([[3,0]]),[np.zeros(3)]),(np.array([[.5,0]]),[np.zeros(3)]),
                     (np.array([[0,1]]),[np.zeros(4)])]:
            with self.assertRaises(ValueError):impute(h,gl,[0])

    def test_reproducible_sampling(self):
        h=np.array([[0,1,2]],np.uint8);gl=[np.log([1,.3,.1,.4,.2,.1])]
        a=impute(h,gl,[0],seed=234);b=impute(h,gl,[0],seed=234)
        np.testing.assert_array_equal(a[0],b[0])


if __name__=='__main__':unittest.main()
