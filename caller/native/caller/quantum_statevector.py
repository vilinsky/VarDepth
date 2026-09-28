"""Exact global-complement symmetry reduction of the existing phase QAOA.

The +1 eigenspace of X^tensor(n) is encoded by n-1 canonical phase bits.
Z0*Zj becomes Zj, Zi*Zj remains Zi*Zj; X0 becomes product(X1..Xn-1).
Keeping that last mixer term is essential: dropping it changes the algorithm.
This reduces classical statevector memory, not evidence of quantum advantage.
"""
import json,time,statistics
from pathlib import Path
import numpy as np

def tapered_circuit(n,edges,angles,measure=False):
    from qiskit import QuantumCircuit
    if n<2 or len(angles)%2:raise ValueError('At least two sites and gamma/beta pairs required')
    m=n-1;qc=QuantumCircuit(m);qc.h(range(m));scale=max([abs(w) for w in edges.values()]+[1.])
    for gamma,beta in np.asarray(angles).reshape(-1,2):
        for (i,j),w in sorted(edges.items()):
            if not 0<=i<j<n:raise ValueError('Canonical site ordering required')
            if i==0:qc.rz(-float(gamma)*w/scale,j-1)
            else:qc.rzz(-float(gamma)*w/scale,i-1,j-1)
        for i in range(m):qc.rx(2*float(beta),i)
        if m==1:qc.rx(2*float(beta),0)
        else:
            qc.h(range(m))
            for i in range(1,m):qc.cx(i-1,i)
            qc.rz(2*float(beta),m-1)
            for i in reversed(range(1,m)):qc.cx(i-1,i)
            qc.h(range(m))
    if measure:qc.measure_all()
    return qc

def canonical_probabilities(full):
    n=int(np.log2(len(full)));codes=np.arange(1<<n,dtype=np.uint64)
    # Qubit zero is the least-significant bit in Qiskit's convention.
    canonical=np.where(codes&1,((1<<n)-1)^codes,codes)>>1
    result=np.zeros(1<<(n-1));np.add.at(result,canonical.astype(np.int64),full)
    return result

def prepare_diagonal(n,edges,taper=True):
    """Dense classical simulator data; exponential allocation is counted."""
    m=n-1 if taper else n
    if not 1<=m<=22:raise ValueError('Classical simulation bounded to 22 statevector qubits')
    codes=np.arange(1<<m,dtype=np.uint64);energy=np.zeros(len(codes));scale=max([abs(w) for w in edges.values()]+[1.])
    for (i,j),w in sorted(edges.items()):
        first=0 if taper and i==0 else (codes>>(i-1 if taper else i))&1
        second=(codes>>(j-1 if taper else j))&1
        energy+=(w/scale)*(first^second)
    return energy

def native_probabilities(n,edges,angles,taper=True,energy=None):
    """Exact dense statevector simulation with fused commuting cost gates.

    The product-X mixer is a complementary-index update; a physical QPU still
    needs the explicit circuit decomposition above. Not a hardware speedup.
    """
    m=n-1 if taper else n
    energy=prepare_diagonal(n,edges,taper) if energy is None else energy
    state=np.full(1<<m,1/np.sqrt(1<<m),dtype=np.complex128)
    for gamma,beta in np.asarray(angles).reshape(-1,2):
        state*=np.exp(-1j*gamma*energy);c=np.cos(beta);s=-1j*np.sin(beta)
        for i in range(m):
            view=state.reshape(-1,2,1<<i);left=view[:,0,:].copy();right=view[:,1,:].copy()
            view[:,0,:]=c*left+s*right;view[:,1,:]=s*left+c*right
        if taper:state=c*state+s*state[::-1]
    return np.abs(state)**2


def simulate(problem,*,seed=17,layers=2,shots=1024,max_evaluations=40,max_qubits=16):
    """Fast exact statevector QAOA in the global-complement symmetry sector."""
    from scipy.optimize import minimize
    problem.validate();n=len(problem.sites)
    if not 2<=n<=max_qubits:raise ValueError('Phase block exceeds site limit')
    if not 1<=layers<=8 or not 1<=shots<=100000 or not 1<=max_evaluations<=2000:raise ValueError('QAOA budget outside limits')
    start=time.perf_counter();edges=problem.edges();energy=prepare_diagonal(n,edges,True);evaluations=0
    def objective(angles):
        nonlocal evaluations
        evaluations+=1
        return float(np.dot(native_probabilities(n,edges,angles,True,energy),energy))
    rng=np.random.default_rng(seed);initial=rng.uniform(-.7,.7,2*layers)
    opt=minimize(objective,initial,method='COBYLA',options={'maxiter':max_evaluations,'rhobeg':.4,'tol':1e-3})
    probabilities=native_probabilities(n,edges,opt.x,True,energy);probabilities/=probabilities.sum()
    draws=rng.choice(len(probabilities),size=shots,p=probabilities);codes,counts=np.unique(draws,return_counts=True)
    merged={tuple([0]+[(int(code)>>i)&1 for i in range(n-1)]):int(count) for code,count in zip(codes,counts)}
    raw=min(merged,key=lambda b:(problem.full_nll(b),b));qc=tapered_circuit(n,edges,opt.x,measure=True)
    gates=qc.count_ops()
    return dict(bits=list(raw),nll=problem.full_nll(raw),seconds=time.perf_counter()-start,
                backend='native-exact-statevector-symmetry-sector',hardware_executed=False,quantum_advantage_claimed=False,
                qubits=n-1,phase_sites=n,canonical_site_zero_fixed=True,layers=layers,shots=shots,
                objective_evaluations=evaluations,angles=opt.x.tolist(),optimizer_success=bool(opt.success),
                optimizer_message=str(opt.message),sampled_unique=len(merged),surrogate='normalized pairwise read-phase Ising',
                rescore='full physical-molecule log likelihood; no genotype update',
                counts={''.join(map(str,b)):v for b,v in merged.items()},circuit_depth=qc.depth(),
                two_qubit_gates=int(gates.get('rzz',0)+gates.get('cx',0)),
                amplitude_storage_bytes=16*(1<<(n-1)),storage_note='Amplitude storage only; simulator has additional temporary arrays and an exponential diagonal')
