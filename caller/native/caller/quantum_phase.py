"""Experimental hybrid QAOA proposal solver for bounded read-phasing problems.

Input: fixed diploid heterozygous SNVs and physical-molecule observations.
Output: orientations only. Never modifies a genotype or a genotype posterior.
The Ising pairwise objective is a surrogate; proposals are re-ranked with the
full molecule likelihood. No quantum speedup or phase calibration is assumed.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
import time
import numpy as np

@dataclass(frozen=True)
class Observation:
    site:int
    allele:int
    error:float

@dataclass
class PhaseProblem:
    sites:list
    fragments:list

    def validate(self):
        if len(self.sites)!=len(set(self.sites)):raise ValueError('Duplicate phase site')
        for fragment in self.fragments:
            seen=set()
            for o in fragment:
                if o.site in seen or not 0<=o.site<len(self.sites):raise ValueError('Duplicate/out-of-range molecule site')
                if o.allele not in (0,1) or not 0<o.error<.5:raise ValueError('Invalid biallelic observation')
                seen.add(o.site)
        return self

    def edges(self):
        self.validate();result={}
        for fragment in self.fragments:
            # A k-site fragment contributes O(k), rather than O(k^2), weight.
            # Still a composite objective, not independent evidence or GP.
            norm=max(1,len(fragment)-1)
            for a,o in enumerate(fragment):
                for v in fragment[a+1:]:
                    i,j=sorted((o.site,v.site));e=o.error+v.error-2*o.error*v.error
                    weight=math.log((1-e)/e)/norm
                    if o.allele!=v.allele:weight=-weight
                    result[i,j]=result.get((i,j),0.)+weight
        return {key:weight for key,weight in result.items() if abs(weight)>1e-12}

    def full_nll(self,bits):
        if len(bits)!=len(self.sites) or any(b not in (0,1) for b in bits):raise ValueError('Invalid orientation')
        value=0.
        for fragment in self.fragments:
            a=b=0.
            for o in fragment:
                p=1-o.error if bits[o.site]==o.allele else o.error
                a+=math.log(p);b+=math.log1p(-p)
            value-=float(np.logaddexp(a,b)-math.log(2.))
        return value

def canonical(bits):
    bits=tuple(int(b) for b in bits)
    return tuple(1-b for b in bits) if bits and bits[0] else bits

def pair_energy(bits,edges):
    return sum(weight*(bits[i]!=bits[j]) for (i,j),weight in edges.items())

def exact(problem,max_sites=18):
    """Independent exhaustive oracle for the full fragment likelihood.

    Exploits only whole-block haplotype label symmetry. Intended for small
    validation/subproblems, not claimed as a scalable chromosome solver.
    """
    problem.validate();n=len(problem.sites)
    if n>max_sites:raise ValueError('Exact solver size limit exceeded')
    if not n:return dict(bits=[],nll=0.,states=1,seconds=0.)
    start=time.perf_counter();codes=np.arange(1<<(n-1),dtype=np.uint64)
    bits=np.zeros((len(codes),n),dtype=np.uint8)
    for i in range(1,n):bits[:,i]=(codes>>(i-1))&1
    energies=np.zeros(len(codes),dtype=np.float64)
    for fragment in problem.fragments:
        a=np.zeros(len(codes));b=a.copy()
        for o in fragment:
            p=np.where(bits[:,o.site]==o.allele,1-o.error,o.error)
            a+=np.log(p);b+=np.log1p(-p)
        energies-=np.logaddexp(a,b)-math.log(2.)
    best=int(np.argmin(energies));ordered=np.sort(energies)
    return dict(bits=bits[best].tolist(),nll=float(energies[best]),states=len(codes),
                next_solution_gap=float(ordered[1]-ordered[0]) if len(ordered)>1 else None,
                seconds=time.perf_counter()-start)

def classical_proposals(problem,seed=17,restarts=8):
    """Bounded coordinate descent using the full molecule likelihood."""
    rng=np.random.default_rng(seed);n=len(problem.sites);result={}
    for k in range(restarts):
        bits=[0]*n if k==0 else rng.integers(0,2,n).tolist();bits=list(canonical(bits))
        score=problem.full_nll(bits)
        for sweep in range(2*n+1):
            improved=False
            for i in range(1,n):
                bits[i]^=1;new=problem.full_nll(bits)
                if new<score-1e-12:score=new;improved=True
                else:bits[i]^=1
            if not improved:break
        result[tuple(bits)]=score
    return sorted(result,key=lambda b:(result[b],b))

def circuit(n,edges,angles,measure=False):
    """QAOA for E=sum w*(1-ZiZj)/2, with a global scale for conditioning."""
    from qiskit import QuantumCircuit
    if len(angles)%2:raise ValueError('QAOA requires gamma/beta pairs')
    qc=QuantumCircuit(n);qc.h(range(n));scale=max([abs(w) for w in edges.values()]+[1.])
    for gamma,beta in np.asarray(angles).reshape(-1,2):
        for (i,j),w in sorted(edges.items()):qc.rzz(-float(gamma)*w/scale,i,j)
        for i in range(n):qc.rx(2*float(beta),i)
    if measure:qc.measure_all()
    return qc

def qaoa_simulator(problem,*,seed=17,layers=2,shots=1024,max_evaluations=40,max_qubits=16):
    """Optimize exact statevector expectation; then perform finite-shot sampling.

    Classical optimization and quantum simulation times are included. The
    returned raw proposal is separated from classical fallback and oracle.
    """
    from qiskit.quantum_info import Statevector
    from qiskit.primitives import StatevectorSampler
    from scipy.optimize import minimize
    problem.validate();n=len(problem.sites)
    if not 2<=n<=max_qubits:raise ValueError('QAOA block must contain 2..max_qubits sites')
    if not 1<=layers<=8 or not 1<=shots<=100000 or not 1<=max_evaluations<=2000:raise ValueError('QAOA budget outside limits')
    start=time.perf_counter();edges=problem.edges();codes=np.arange(1<<n,dtype=np.uint64)
    energies=np.zeros(len(codes));scale=max([abs(w) for w in edges.values()]+[1.])
    for (i,j),w in edges.items():energies+=w*(((codes>>i)^(codes>>j))&1)/scale
    evaluations=0
    def objective(angles):
        nonlocal evaluations
        evaluations+=1;state=Statevector.from_instruction(circuit(n,edges,angles))
        return float(np.dot(np.abs(state.data)**2,energies))
    rng=np.random.default_rng(seed);initial=rng.uniform(-.7,.7,2*layers)
    opt=minimize(objective,initial,method='COBYLA',options={'maxiter':max_evaluations,'rhobeg':.4,'tol':1e-3})
    qc=circuit(n,edges,opt.x,measure=True)
    counts=StatevectorSampler(seed=seed).run([qc],shots=shots).result()[0].data.meas.get_counts()
    merged={}
    for text,count in counts.items():
        bits=canonical([int(x) for x in text[::-1]]) # Qiskit little-endian qubits.
        merged[bits]=merged.get(bits,0)+int(count)
    ranked=sorted(merged,key=lambda b:(problem.full_nll(b),b));raw=ranked[0]
    return dict(bits=list(raw),nll=problem.full_nll(raw),seconds=time.perf_counter()-start,
                backend='qiskit-statevector-simulator',hardware_executed=False,quantum_advantage_claimed=False,
                qubits=n,layers=layers,shots=shots,objective_evaluations=evaluations,angles=opt.x.tolist(),
                optimizer_success=bool(opt.success),optimizer_message=str(opt.message),
                sampled_unique=len(merged),surrogate='normalized pairwise read-phase Ising',
                rescore='full physical-molecule log likelihood; no genotype update',
                counts={''.join(map(str,b)):v for b,v in merged.items()},
                circuit_depth=qc.depth(),two_qubit_gates=int(qc.count_ops().get('rzz',0)))

def hybrid(problem,*,seed=17,layers=2,shots=1024,max_evaluations=40,oracle=True,simulator="qiskit"):
    start=time.perf_counter();cstart=time.perf_counter()
    candidates=classical_proposals(problem,seed=seed);cbits=candidates[0]
    classical=dict(bits=list(cbits),nll=problem.full_nll(cbits),seconds=time.perf_counter()-cstart)
    if simulator not in ('qiskit','native'):raise ValueError('Unknown quantum simulator')
    if simulator=='native':
        from quantum_statevector import simulate
        quantum=simulate(problem,seed=seed,layers=layers,shots=shots,max_evaluations=max_evaluations)
    else:quantum=qaoa_simulator(problem,seed=seed,layers=layers,shots=shots,max_evaluations=max_evaluations)
    qbits=tuple(quantum['bits']);chosen=min([cbits,qbits],key=lambda b:(problem.full_nll(b),b))
    result=dict(bits=list(chosen),nll=problem.full_nll(chosen),classical=classical,quantum=quantum,
                selected='quantum-proposal' if problem.full_nll(qbits)<classical['nll']-1e-10 else 'classical-fallback',
                seconds=time.perf_counter()-start,phase_confidence_calibrated=False)
    if oracle:
        result['exact']=exact(problem)
        result['quantum_gap_to_exact']=quantum['nll']-result['exact']['nll']
        result['hybrid_gap_to_exact']=result['nll']-result['exact']['nll']
    return result

def export_qasm(problem,angles,path,*,taper=False):
    from pathlib import Path
    from qiskit import qasm3
    if taper:
        from quantum_statevector import tapered_circuit
        qc=tapered_circuit(len(problem.sites),problem.edges(),angles,measure=True)
    else:qc=circuit(len(problem.sites),problem.edges(),angles,measure=True)
    Path(path).write_text(qasm3.dumps(qc))

def run_sampler(problem,angles,sampler,*,pass_manager=None,shots=1024):
    """Adapter for caller-provided SamplerV2 (real QPU or simulator).

    Authentication, provider choice and submitting the job are explicit in the
    calling program. No credentials are read and no QPU job is sent by default.
    ISA transpilation is required for hardware; supplied by pass_manager.
    """
    if not 1<=shots<=100000:raise ValueError('Shot budget outside limits')
    qc=circuit(len(problem.sites),problem.edges(),angles,measure=True)
    if pass_manager is not None:qc=pass_manager.run(qc)
    job=sampler.run([qc],shots=shots);counts=job.result()[0].data.meas.get_counts()
    bits=min((canonical([int(x) for x in text[::-1]]) for text in counts),key=lambda b:(problem.full_nll(b),b))
    return dict(bits=list(bits),nll=problem.full_nll(bits),counts=counts,
                job_id=job.job_id() if hasattr(job,'job_id') else None,shots=shots,
                backend='external-SamplerV2',hardware_executed='must be attested by caller/provider')
