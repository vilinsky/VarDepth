#!/usr/bin/env python3
"""Synthetic throughput profile; never a genomic accuracy benchmark."""
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from model import NeuralCaller,ModelConfig,tensors

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--steps',type=int,default=30);p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.manual_seed(21)
    n=a.batch_size;rng=np.random.default_rng(21)
    counts=rng.integers(2,5,n);am=np.arange(12)[None,:]<counts[:,None];gm=np.arange(78)[None,:]<(counts*(counts+1)//2)[:,None]
    gp=rng.random((n,78))*gm;gp/=gp.sum(1,keepdims=True)
    x={'allele_sequences':rng.integers(1,5,(n,12,256),dtype=np.int8)*am[:,:,None],
       'allele_mask':am,'genotype_mask':gm,'fragment_features':rng.normal(size=(n,64,20)).astype('f'),
       'fragment_mask':rng.random((n,64))>.3,'base_gp':gp.astype('f'),
       'read_pl':(-10*np.log10(np.maximum(gp,1e-30))).astype('f'),
       'depth':rng.integers(0,101,n).astype('f'),'allele_features':rng.random((n,12,8)).astype('f'),
       'cohort_context':rng.random((n,12,6)).astype('f'),'context_strength':rng.random(n).astype('f')}
    model=NeuralCaller(ModelConfig()).to(a.device);batch=tensors(x,a.device);labels=torch.ones(n,dtype=torch.long,device=a.device)
    optimizer=torch.optim.AdamW(model.parameters());times=[]
    for i in range(a.steps+5):
        if a.device=='cuda':torch.cuda.synchronize()
        start=time.perf_counter();optimizer.zero_grad();o=model(batch,return_aux=True)
        loss=torch.nn.functional.cross_entropy(o['logits'],labels)+.15*torch.nn.functional.cross_entropy(o['read_logits'],labels)
        loss.backward();optimizer.step()
        if a.device=='cuda':torch.cuda.synchronize()
        if i>=5:times.append(time.perf_counter()-start)
    report={'scope':'Synthetic training throughput only; excludes BAM extraction, IO, evaluation, and realistic candidate distribution.',
            'torch':str(torch.__version__),'device':torch.cuda.get_device_name(0) if a.device=='cuda' else a.device,
            'parameters':sum(p.numel() for p in model.parameters()),'batch_size':n,'steps':len(times),
            'median_seconds_per_batch':float(np.median(times)),'examples_per_second':n/float(np.mean(times)),
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else None}
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
