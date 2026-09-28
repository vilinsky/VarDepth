"""Strong classical controls for quantum phase experiments.

The objective is the full physical-molecule likelihood, without a pairwise
surrogate. Exact elimination exploits graph structure, with an explicit bound
on intermediate factor width. Used for research controls, not genotype calling.
"""
import itertools,math,time
import numpy as np

def factors(problem):
    problem.validate();out=[]
    for fragment in problem.fragments:
        scope=tuple(sorted(o.site for o in fragment));n=len(scope)
        if n>20:raise ValueError('Fragment factor exceeds explicit 20-site budget')
        codes=np.arange(1<<n,dtype=np.uint64);a=np.zeros(len(codes));b=a.copy()
        for o in fragment:
            bit=(codes>>(n-1-scope.index(o.site)))&1
            p=np.where(bit==o.allele,1-o.error,o.error)
            a+=np.log(p);b+=np.log1p(-p)
        out.append((scope,(-np.logaddexp(a,b)+math.log(2.)).reshape((2,)*n)))
    return out

def eliminate(problem,max_width=20,fixed=None):
    """Exact min-sum elimination; canonicalize the first phase bit to zero."""
    if not 1<=max_width<=24:raise ValueError('Exact factor-width budget must be 1..24')
    start=time.perf_counter();fs=factors(problem);n=len(problem.sites);active=[]
    fixed=dict(fixed or {})
    if any(not isinstance(i,int) or not 0<=i<n or bit not in (0,1) for i,bit in fixed.items()):
        raise ValueError('Invalid fixed phase bits')
    if fixed.get(0,0)!=0:raise ValueError('First phase bit is canonically zero')
    fixed[0]=0
    for scope,table in fs:
        for variable,bit in sorted(fixed.items()):
            if variable in scope:
                axis=scope.index(variable);table=np.take(table,bit,axis=axis);scope=tuple(v for v in scope if v!=variable)
        active.append((scope,table))
    remaining=set(range(n))-set(fixed);trace=[];largest=0;total_states=0
    while remaining:
        adjacency={i:set() for i in remaining}
        for scope,table in active:
            for i in scope:adjacency[i].update(j for j in scope if j!=i)
        def cost(i):
            neighbors=sorted(adjacency[i]);fill=sum(b not in adjacency[a] for a,b in itertools.combinations(neighbors,2))
            return fill,len(neighbors),i
        variable=min(remaining,key=cost);bucket=[x for x in active if variable in x[0]]
        scope=tuple(sorted(set().union(*(set(s) for s,t in bucket)))) if bucket else (variable,)
        if len(scope)>max_width:
            return dict(solved=False,reason='factor width budget',next_width=len(scope),max_width=largest,
                        seconds=time.perf_counter()-start,total_factor_states=total_states)
        joint=np.zeros((2,)*len(scope));largest=max(largest,len(scope));total_states+=joint.size
        for source,table in bucket:
            shape=[2 if i in source else 1 for i in scope]
            joint+=table.reshape(shape)
        axis=scope.index(variable);arg=joint.argmin(axis=axis).astype(np.uint8);message=joint.min(axis=axis)
        residual=tuple(i for i in scope if i!=variable);trace.append((variable,residual,arg))
        active=[x for x in active if variable not in x[0]];active.append((residual,message));remaining.remove(variable)
    bits=[fixed.get(i,0) for i in range(n)]
    for variable,scope,arg in reversed(trace):bits[variable]=int(arg[tuple(bits[i] for i in scope)])
    value=sum(float(table) for scope,table in active)
    checked=problem.full_nll(bits)
    if abs(value-checked)>1e-8*max(1,abs(checked)):raise RuntimeError('Elimination reconstruction failed')
    return dict(solved=True,bits=bits,nll=checked,seconds=time.perf_counter()-start,max_width=largest,
                total_factor_states=total_states,largest_factor_bytes=8*(1<<largest),
                scope='Exact full-molecule likelihood; symmetry-fixed, bounded min-fill variable elimination')

def eliminate_unique(problem,max_width=20):
    """Certify the runner-up by excluding each bit of the optimum in turn.

    Every distinct canonical assignment disagrees at one non-fixed bit, so the
    minimum of these constrained optima is exactly the second-best energy.
    The extra solves are included in reported time; ties must abstain in VCF.
    """
    start=time.perf_counter();best=eliminate(problem,max_width);solves=[best]
    if not best['solved']:return dict(best,uniqueness_certified=False)
    second=float('inf')
    for i in range(1,len(best['bits'])):
        other=eliminate(problem,max_width,{i:1-best['bits'][i]});solves.append(other)
        if not other['solved']:
            return dict(best,solved=False,uniqueness_certified=False,reason='Runner-up width budget',seconds=time.perf_counter()-start)
        second=min(second,other['nll'])
    return dict(best,seconds=time.perf_counter()-start,next_solution_gap=second-best['nll'],
        uniqueness_certified=True,exact_solves=len(solves),max_width=max(s['max_width'] for s in solves),
        total_factor_states=sum(s['total_factor_states'] for s in solves))

def anneal(problem,seed=17,restarts=16,sweeps=100):
    """Reproducible factor-local simulated annealing plus coordinate polishing."""
    if restarts<1 or sweeps<1:raise ValueError("Positive search budgets required")
    start=time.perf_counter();n=len(problem.sites);fs=factors(problem);rng=np.random.default_rng(seed)
    affected=[[] for _ in range(n)]
    for f,(scope,table) in enumerate(fs):
        for site in scope:affected[site].append((f,1<<(len(scope)-1-scope.index(site))))
    tables=[t.ravel() for s,t in fs];best=None;best_value=math.inf;proposals=0
    temperature=max([float(t.max()-t.min())/max(1,len(s)) for s,t in fs]+[1.])
    for restart in range(restarts):
        bits=rng.integers(0,2,n).tolist()
        if n:bits[0]=0
        codes=[sum(bits[v]<<(len(scope)-1-j) for j,v in enumerate(scope)) for scope,t in fs]
        value=sum(float(table[c]) for table,c in zip(tables,codes))
        def delta(site):return sum(float(tables[f][codes[f]^mask]-tables[f][codes[f]]) for f,mask in affected[site])
        def flip(site,change):
            nonlocal value
            bits[site]^=1;value+=change
            for f,mask in affected[site]:codes[f]^=mask
        for sweep in range(sweeps):
            temp=temperature*(.001**(sweep/max(1,sweeps-1)))
            for site in rng.permutation(np.arange(1,n)):
                site=int(site);change=delta(site);proposals+=1
                if change<=0 or rng.random()<math.exp(-change/temp):flip(site,change)
        for sweep in range(2*n+1):
            changes=0
            for site in range(1,n):
                change=delta(site);proposals+=1
                if change < -1e-12:flip(site,change);changes+=1
            if not changes:break
        value=problem.full_nll(bits)
        if value<best_value-1e-12 or (abs(value-best_value)<=1e-12 and tuple(bits)<tuple(best or bits)):
            best=list(bits);best_value=value
    return dict(bits=best,nll=best_value,seconds=time.perf_counter()-start,proposals=proposals,
                restarts=restarts,sweeps=sweeps,objective='full molecule likelihood')
