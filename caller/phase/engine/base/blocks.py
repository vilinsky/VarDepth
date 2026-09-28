"""Bounded, exact local two-SNV fragment factors for a copying HMM.
A fragment is marginalized over its two chromosome origins once per block.
Within a block, mate observations share that origin. Across blocks the fragment
origin is approximated as independent; this is explicitly not full QUILT-like
whole-fragment inference. Indels use the read-only PairHMM site likelihood.
"""
import ctypes,itertools,math
import numpy as np

def blocks_for(variants,max_distance=500):
    blocks=[];i=0
    while i<len(variants):
        v=variants[i]
        if i+1<len(variants) and len(v[1])==len(v[2])==len(variants[i+1][1])==len(variants[i+1][2])==1 and 0<variants[i+1][0]-v[0]<=max_distance:
            blocks.append([i,i+1]);i+=2
        else:blocks.append([i]);i+=1
    return blocks

def factors(variants,hap,gl,mu,observations,max_distance=500):
    blocks=blocks_for(variants,max_distance);m,k=hap.shape;kk=k*k
    emissions=np.zeros((len(blocks),kk));eg=np.zeros((m,kk,3));phase_n=np.zeros((m,kk,2));block_of=np.empty(m,dtype=np.int32);logscale=0
    for t,indices in enumerate(blocks):
        for j in indices:block_of[j]=t
        if len(indices)==1:
            j=indices[0];x=mu[j]+(1-2*mu[j])*hap[j];a=x[:,None];b=x[None,:]
            prior=np.array([(1-a)*(1-b),a*(1-b)+(1-a)*b,a*b]).reshape(3,kk).T
            l=np.exp(gl[j]-max(gl[j]));weights=prior*l;eg[j]=weights;emissions[t]=weights.sum(axis=1)
            phase_n[j,:,0]=(a*(1-b)*l[1]+a*b*l[2]).reshape(-1);phase_n[j,:,1]=((1-a)*b*l[1]+a*b*l[2]).reshape(-1)
            logscale+=max(gl[j]);continue
        fragments={}
        for local,j in enumerate(indices):
            for key,obs in observations.get(j,{}).items():fragments.setdefault(key,{})[local]=obs
        configs=[]
        for a in itertools.product([0,1],repeat=2):
            for b in itertools.product([0,1],repeat=2):
                ll=0.
                for calls in fragments.values():
                    la=lb=1.
                    for local,(base,bq,mq) in calls.items():
                        ref,alt=variants[indices[local]][1:];e=10**(-min(bq,60)/10);em=10**(-min(mq,60)/10)
                        def p(allele):return (1-em)*(1-e if base==(alt if allele else ref) else e/3)+em/4
                        la*=p(a[local]);lb*=p(b[local])
                    ll+=math.log(max(1e-300,(la+lb)/2))
                configs.append((a,b,ll))
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

def block_infer(lib,variants,hap,gl,mu,observations,no_ld=False,artifact_prior=None,recombination=None,indel_artifact_prior=None):
    if artifact_prior is None:blocks,e,eg,pn,bof,logscale=factors(variants,hap,gl,mu,observations)
    else:
        from artifact import factors as artifact_factors
        blocks,e,eg,pn,bof,logscale=artifact_factors(variants,hap,gl,mu,observations,artifact_prior=artifact_prior,indel_artifact_prior=indel_artifact_prior)
    pos=np.array([variants[x[0]][0] for x in blocks]);dist=np.diff(pos,prepend=pos[0]);r=np.ones(len(blocks)) if no_ld else np.clip(1-np.exp(-dist*1e-6),1e-8,.5)
    if recombination:
        from genetic_map import transitions
        r=transitions(pos,no_ld=no_ld,**recombination)
    gp=np.empty_like(gl);sp=np.empty_like(e);p=lambda x:x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    lib.block_hmm.argtypes=[ctypes.c_int]*3+[ctypes.POINTER(ctypes.c_double)]*2+[ctypes.POINTER(ctypes.c_int)]+[ctypes.POINTER(ctypes.c_double)]*3;lib.block_hmm.restype=ctypes.c_double
    ll=lib.block_hmm(len(blocks),len(variants),hap.shape[1],p(e),p(eg),bof.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),p(r),p(gp),p(sp))
    phase=np.zeros((len(variants),2))
    for j in range(len(variants)):
        t=bof[j];s=int(np.argmax(sp[t]));phase[j]=pn[j,s]/max(1e-300,e[t,s])
    return gp,phase,{'log_marginal_likelihood':ll+logscale,'blocks':len(blocks),'two_snv_blocks':sum(len(x)==2 for x in blocks),'fragment_linked_blocks':sum(bool(set(observations.get(x[0],{}))&set(observations.get(x[1],{}))) for x in blocks if len(x)==2)}
