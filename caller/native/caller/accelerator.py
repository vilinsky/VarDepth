"""Bounded optional CUDA likelihood batches; CPU remains the default.

The default CUDA implementation evaluates the same model in float64.
Optional mixed mode uses float32 DP, double log scales, and strict fallback
for long pairs, tiny terminal mass, nonfinite or extreme log likelihoods.
This is an empirically validated approximation, not a formal error bound.
Transcendental rounding can differ; parity must be measured, not assumed.
Explicit CUDA requests fail clearly if unavailable. Auto dispatch is opt-in.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np

HERE=Path(__file__).resolve().parent
_ENGINE=None
_ENGINE_KEY=None
_ERROR_LUT=np.asarray([math.pow(10.,-min(60,v)/10.) for v in range(256)],dtype=np.float64)

def validate_pairs(pairs):
    result=[];checked_sequences=set();checked_qualities={}
    for sequence, qualities, haplotype in pairs:
        if len(sequence)!=len(qualities): raise ValueError('Read/quality lengths differ')
        for dna in (sequence,haplotype):
            if dna not in checked_sequences:
                if not isinstance(dna,str) or not set(dna)<=set('ACGTN'):raise ValueError('Uppercase DNA required')
                checked_sequences.add(dna)
        key=id(qualities)
        if key not in checked_qualities:
            q=np.frombuffer(qualities,np.uint8) if isinstance(qualities,bytes) else np.asarray(qualities)
            if q.ndim!=1 or np.any(~np.isfinite(q)) or np.any(q<0) or np.any(q>255) or np.any(q!=np.floor(q)):
                raise ValueError('Base qualities must be integer Phred scores in [0,255]')
            checked_qualities[key]=(qualities,np.asarray(q,dtype=np.uint8)) # Keep the object alive; IDs cannot be recycled.
        result.append((sequence,checked_qualities[key][1],haplotype))
    return result

def cpu_pairs(pairs, anchored=False):
    pairs=validate_pairs(pairs)
    if anchored:
        from indel_refine import kernel
        fn=kernel().anchored_pairhmm
    else:
        from whole_pairhmm import library
        lib=library();fn=lib.read_haplotype
        fn.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_ubyte),ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_double,ctypes.c_double]
        fn.restype=ctypes.c_double
    return np.asarray([fn(s.encode(),q.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),len(s),h.encode(),len(h),.001,.1)
                       for s,q,h in pairs],dtype=np.float64)

def cpu_simd_pairs(pairs):
    """Exact semiglobal fallback, batching haplotypes that share one read."""
    from whole_pairhmm import Prepared
    pairs=validate_pairs(pairs);groups={};out=np.empty(len(pairs),np.float64)
    for i,(s,q,h) in enumerate(pairs):groups.setdefault((s,q.tobytes()),[]).append(i)
    for (s,q),ids in groups.items():
        out[ids]=Prepared([pairs[i][2] for i in ids])(s,np.frombuffer(q,np.uint8))
    return out

def pack_inputs(pairs,selected,nr,nh,*,pair_major=False,encoded=None,qerrors=None):
    """Pack directly in the CUDA kernel's memory order, avoiding a transpose."""
    encoded={} if encoded is None else encoded;qerrors={} if qerrors is None else qerrors
    n=len(selected)
    reads=np.zeros((n,nr) if pair_major else (nr,n),np.uint8)
    haps=np.zeros((n,nh) if pair_major else (nh,n),np.uint8)
    errors=np.zeros(reads.shape,np.float64);rl=np.empty(n,np.int32);hl=np.empty(n,np.int32)
    rv=reads if pair_major else reads.T;hv=haps if pair_major else haps.T;ev=errors if pair_major else errors.T
    for j,idx in enumerate(selected):
        s,q,h=pairs[idx];rl[j]=len(s);hl[j]=len(h)
        if s not in encoded:encoded[s]=np.frombuffer(s.encode(),np.uint8)
        if h not in encoded:encoded[h]=np.frombuffer(h.encode(),np.uint8)
        if id(q) not in qerrors:qerrors[id(q)]=_ERROR_LUT[q]
        rv[j,:len(s)]=encoded[s];hv[j,:len(h)]=encoded[h];ev[j,:len(s)]=qerrors[id(q)]
    return [reads,haps,errors,rl,hl]

class CudaPairHMM:
    def __init__(self, device=0, memory_mb=256, precision=None, strategy=None):
        strategy=strategy or os.environ.get("VARDEPTH_CUDA_KERNEL","pair")
        if strategy not in ("pair","warp"):raise ValueError("CUDA kernel must be pair or warp")
        self.strategy=strategy
        self.hap_bucket=int(os.environ.get("VARDEPTH_WARP_HAP_BUCKET","128"))
        if self.hap_bucket<32:raise ValueError("Warp haplotype bucket must be >=32")
        precision=precision or os.environ.get("VARDEPTH_CUDA_PRECISION","float64")
        if precision not in ("float64","mixed"):raise ValueError("CUDA precision must be float64 or mixed")
        self.precision=precision;self._strict=None
        if memory_mb<8: raise ValueError('CUDA workspace budget must be at least 8 MiB')
        try:
            import cupy as cp
            cp.cuda.Device(device).use()
            props=cp.cuda.runtime.getDeviceProperties(device)
        except (ImportError,RuntimeError) as exc:
            raise RuntimeError('CUDA PairHMM requires CuPy and an accessible NVIDIA GPU') from exc
        self.cp=cp;self.device=device;self.memory_bytes=int(memory_mb)*1024**2
        source=(HERE/('pairhmm_warp.cu' if strategy=='warp' else 'pairhmm_cuda.cu')).read_text()
        self.kernel=cp.RawKernel(source,'pairhmm',options=('--std=c++17','--fmad=false')+(('-DVARDEPTH_MIXED',) if precision=='mixed' else ()))
        self.metadata=dict(backend='cuda',strategy=strategy,hap_bucket=self.hap_bucket if strategy=='warp' else 32,device=device,gpu=props['name'].decode() if isinstance(props['name'],bytes) else props['name'],
                           compute_capability=[props['major'],props['minor']],cupy=cp.__version__,
                           driver=cp.cuda.runtime.driverGetVersion(),precision=precision,fmad=False,
                           kernel_sha256=hashlib.sha256(source.encode()).hexdigest(),memory_budget_bytes=self.memory_bytes)

    def score(self,pairs,anchored=False):
        start=time.perf_counter();pairs=validate_pairs(pairs);cp=self.cp
        result=np.full(len(pairs),-np.inf,dtype=np.float64)
        # Similar shapes in a warp reduce padding and divergent loop tails.
        groups={}
        for idx,(s,q,h) in enumerate(pairs):
            if s and h:
                key=((len(s)+127)//128,(len(h)+self.hap_bucket-1)//self.hap_bucket) if self.strategy=='warp' else ((len(s)+31)//32,(len(h)+31)//32)
                groups.setdefault(key,[]).append(idx)
        peak=0;batches=0;kernel_ms=0.;packing_seconds=0.;encoded={};qerrors={}
        with cp.cuda.Device(self.device):
            for ids in groups.values():
                nr=max(len(pairs[i][0]) for i in ids);nh=max(len(pairs[i][2]) for i in ids)
                bytes_per_pair=3*(nh+1)*(4 if self.precision=="mixed" else 8)+9*nr+nh+24
                if bytes_per_pair>self.memory_bytes:raise ValueError('One pair exceeds CUDA workspace budget')
                step=max(1,self.memory_bytes//bytes_per_pair)
                for begin in range(0,len(ids),step):
                    selected=ids[begin:begin+step];n=len(selected)
                    pack_start=time.perf_counter()
                    packed=pack_inputs(pairs,selected,nr,nh,pair_major=self.strategy=='warp',encoded=encoded,qerrors=qerrors)
                    packing_seconds+=time.perf_counter()-pack_start
                    gpu_inputs=[cp.asarray(x) for x in packed]
                    workspace=cp.empty(3*(nh+1)*n,dtype=cp.float32 if self.precision=="mixed" else cp.float64);out=cp.empty(n,dtype=cp.float64)
                    begin_event=cp.cuda.Event();end_event=cp.cuda.Event();begin_event.record()
                    if self.strategy=='warp':
                        self.kernel(((n+3)//4,),(128,),(*gpu_inputs,np.int32(n),np.int32(nh),np.int32(nr),np.int32(anchored),workspace,out))
                    else:
                        self.kernel(((n+127)//128,),(128,),(*gpu_inputs,np.int32(n),np.int32(nh),np.int32(anchored),workspace,out))
                    end_event.record()
                    result[selected]=cp.asnumpy(out) # Synchronization and D2H are part of timing.
                    kernel_ms+=float(cp.cuda.get_elapsed_time(begin_event,end_event))
                    batches+=1;peak=max(peak,bytes_per_pair*n)
                    del gpu_inputs,workspace,out
        fallback=[];fallback_seconds=0.;fallback_backend=None
        if self.precision=='mixed':
            fallback=[i for i,(s,q,h) in enumerate(pairs) if s and h and
                      (len(s)>512 or len(h)>1024 or not np.isfinite(result[i]) or abs(result[i])>80)]
            if fallback:
                selected=[pairs[i] for i in fallback]
                # Measured on sparse public work: SIMD avoids underfilled GPU
                # launches. The bound limits serial CPU work, not precision.
                if not anchored and sum(len(s)*len(h) for s,q,h in selected)<=50_000_000:
                    begin=time.perf_counter();result[fallback]=cpu_simd_pairs(selected)
                    fallback_seconds=time.perf_counter()-begin;fallback_backend='cpu_simd_float64'
                else:
                    if self._strict is None:self._strict=CudaPairHMM(self.device,self.memory_bytes//1024**2,precision='float64',strategy=self.strategy)
                    result[fallback]=self._strict.score(selected,anchored)
                    fallback_seconds=self._strict.last['seconds'];fallback_backend='cuda_float64'
                    kernel_ms+=1000*self._strict.last['kernel_seconds'];packing_seconds+=self._strict.last['host_packing_seconds']
                    batches+=self._strict.last['batches'];peak=max(peak,self._strict.last['live_array_bytes_upper_bound'])
        self.last=dict(**self.metadata,float64_fallback_backend=fallback_backend,float64_fallback_pairs=len(fallback),float64_fallback_seconds=fallback_seconds,pairs=len(pairs),dp_cells=sum(len(s)*len(h) for s,q,h in pairs),
                       batches=batches,live_array_bytes_upper_bound=peak,seconds=time.perf_counter()-start,
                       includes_packing_transfers_and_synchronization=True,anchored=anchored,
                       kernel_seconds=kernel_ms/1000.,host_packing_seconds=packing_seconds)
        return result

def backend():
    value=os.environ.get('VARDEPTH_PAIRHMM_BACKEND','cpu')
    if value not in ('cpu','cuda','auto'): raise ValueError('Invalid PairHMM backend')
    return value

def score_pairs(pairs,anchored=False,selection=None):
    global _ENGINE,_ENGINE_KEY
    choice=selection or backend();pairs=list(pairs)
    if choice not in ('cpu','cuda','auto'):raise ValueError('Invalid PairHMM backend')
    if not pairs:return np.empty(0,np.float64)
    cells=sum(len(s)*len(h) for s,q,h in pairs)
    if choice=='cpu' or (choice=='auto' and cells<2_000_000):return cpu_pairs(pairs,anchored)
    key=(int(os.environ.get('VARDEPTH_CUDA_DEVICE','0')),int(os.environ.get('VARDEPTH_CUDA_MEMORY_MB','256')),os.environ.get('VARDEPTH_CUDA_PRECISION','float64'),os.environ.get('VARDEPTH_CUDA_KERNEL','pair'),int(os.environ.get('VARDEPTH_WARP_HAP_BUCKET','128')))
    if _ENGINE is None or _ENGINE_KEY!=key:
        try:_ENGINE=CudaPairHMM(*key[:4]);_ENGINE_KEY=key
        except RuntimeError:
            if choice=='auto':return cpu_pairs(pairs,anchored)
            raise
    result=_ENGINE.score(pairs,anchored)
    telemetry=os.environ.get('VARDEPTH_ACCELERATOR_LOG')
    if telemetry:
        import fcntl
        with open(telemetry,'a') as stream:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
            stream.write(json.dumps(_ENGINE.last)+'\n')
    return result

def score_many(sequences,qualities,haplotypes,selection=None):
    if len(sequences)!=len(qualities):raise ValueError('Sequence/quality batch lengths differ')
    return score_pairs([(s,q,h) for s,q in zip(sequences,qualities) for h in haplotypes],selection=selection).reshape(len(sequences),len(haplotypes))
