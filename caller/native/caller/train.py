#!/usr/bin/env python3
"""Train on independently labelled public features; restart at saved batch.

Dataset JSON: {"role":"train|validation", "entries":[
  {"features":"path.npz", "labels":"labels.npz", "samples":["HG001"]}]}
Labels NPZ contains `labels` in row order, -1 for unlabelled/unresolvable rows.
Feature extraction never reads truth. Test datasets are rejected here.
"""
import argparse
from dataclasses import asdict
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch import nn
from model import NeuralCaller,ModelConfig,MODEL_VERSION,tensors

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def dataset(path,role):
    p=Path(path).resolve();d=json.loads(p.read_text())
    if d.get('role')!=role:raise ValueError('Dataset role must be '+role)
    if not d.get('entries'):raise ValueError('Empty dataset')
    entries=[]
    for item in d['entries']:
        e=dict(item)
        for key in ['features','labels']:
            q=Path(e[key]);q=q if q.is_absolute() else p.parent/q
            e[key]=str(q.resolve());e[key+'_sha256']=sha(q)
        entries.append(e)
    return {'path':str(p),'manifest_sha256':sha(p),'role':role,'entries':entries,
            'split':d.get('split',{}),'samples':d.get('samples',[])}

def load_entry(entry):
    with np.load(entry['features'],allow_pickle=False) as z:x={k:z[k] for k in z.files}
    with np.load(entry['labels'],allow_pickle=False) as z:y=z['labels'].astype(np.int64)
    if len(y)!=len(x['depth']):raise ValueError('Label/feature row mismatch')
    valid=np.flatnonzero(y>=0)
    if len(valid) and (np.any(y[valid]>=78) or not x['genotype_mask'][valid,y[valid]].all()):
        raise ValueError('Truth label is not a valid candidate genotype')
    return x,y,valid

def batches(data,batch_size,epoch,seed,shuffle=True,resume_entry=0,resume_batch=0):
    order=list(range(len(data['entries'])))
    if shuffle:random.Random(seed+epoch*100003).shuffle(order)
    for ordinal,index in enumerate(order):
        if ordinal<resume_entry:continue
        entry=data['entries'][index];x,y,valid=load_entry(entry)
        if shuffle:np.random.default_rng(seed+epoch*100003+index).shuffle(valid)
        for offset in range(0,len(valid),batch_size):
            number=offset//batch_size
            if ordinal==resume_entry and number<resume_batch:continue
            ids=valid[offset:offset+batch_size]
            yield ({k:v[ids] for k,v in x.items() if isinstance(v,np.ndarray) and len(v.shape)>0 and len(v)==len(y)},
                   y[ids],ordinal,number,len(order))

@torch.inference_mode()
def evaluate(model,data,batch_size,device):
    model.eval();n=correct=base_correct=0;loss=base_loss=0.;errors_nonref=base_errors_nonref=0
    nonref_total=nonref_tp=nonref_fp=base_nonref_tp=base_nonref_fp=0
    nonref_loss=ref_loss=0.
    for x,y,*_ in batches(data,batch_size,0,0,False):
        b=tensors(x,device);labels=torch.as_tensor(y,device=device)
        logits=model(b);lp=torch.log_softmax(logits,1);pred=lp.argmax(1)
        base=b['base_gp'].clamp_min(1e-30);bi=base.argmax(1)
        rows=torch.arange(len(y),device=device)
        loss+=float(-lp[rows,labels].sum());base_loss+=float(-base[rows,labels].log().sum())
        correct+=int((pred==labels).sum());base_correct+=int((bi==labels).sum());n+=len(y)
        nonref=labels!=0
        nonref_total+=int(nonref.sum())
        nonref_tp+=int(((pred==labels)&nonref).sum());nonref_fp+=int(((pred!=labels)&(pred!=0)).sum())
        base_nonref_tp+=int(((bi==labels)&nonref).sum());base_nonref_fp+=int(((bi!=labels)&(bi!=0)).sum())
        nonref_loss+=float(-lp[rows[nonref],labels[nonref]].sum())
        ref_loss+=float(-lp[rows[~nonref],labels[~nonref]].sum())
        errors_nonref+=int(((pred!=labels)&((labels!=0)|(pred!=0))).sum())
        base_errors_nonref+=int(((bi!=labels)&((labels!=0)|(bi!=0))).sum())
        # Allele length feature is specified by the extractor. Type-specific
        # full-truth FN still belongs to the external VCF evaluator.
    if not n:raise ValueError('No labelled validation examples')
    return {'labelled_candidates':n,'candidate_accuracy':correct/n,'candidate_errors':n-correct,
            'nll':loss/n,'base_candidate_accuracy':base_correct/n,'base_candidate_errors':n-base_correct,
            'base_nll':base_loss/n,'nonreference_discordance_count':errors_nonref,
            'base_nonreference_discordance_count':base_errors_nonref,
            'truth_nonref_candidates':nonref_total,'nonref_tp':nonref_tp,'nonref_fp':nonref_fp,'nonref_fn':nonref_total-nonref_tp,
            'candidate_nonref_f1':2*nonref_tp/max(1,nonref_tp+nonref_total+nonref_fp),
            'base_candidate_nonref_f1':2*base_nonref_tp/max(1,base_nonref_tp+nonref_total+base_nonref_fp),
            'nonref_nll':nonref_loss/nonref_total if nonref_total else None,
            'ref_nll':ref_loss/(n-nonref_total) if n>nonref_total else None,
            'scope':'Candidate-conditional validation for checkpoint selection; not full-discovery F1.'}

def atomic_save(obj,path):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.partial');torch.save(obj,tmp);tmp.replace(p)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=20260906);p.add_argument('--hidden',type=int,default=48)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--resume',action='store_true')
    p.add_argument('--checkpoint-steps',type=int,default=100);p.add_argument('--max-steps',type=int)
    p.add_argument('--prior-dropout',type=float,default=.1);p.add_argument('--cohort-dropout',type=float,default=.35)
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1 or a.threads<1 or a.checkpoint_steps<1 or (a.max_steps is not None and a.max_steps<1):raise ValueError('Invalid training dimensions')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    # Deterministic algorithms expose unsupported kernels instead of silently
    # claiming exact restart reproducibility.
    torch.use_deterministic_algorithms(True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    train=dataset(a.train,'train');val=dataset(a.validation,'validation')
    config=ModelConfig(hidden=a.hidden,prior_dropout=a.prior_dropout,cohort_dropout=a.cohort_dropout)
    model=NeuralCaller(config).to(a.device);optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    source={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    identity={'train':train,'validation':val,'config':asdict(config),'source_sha256':source,
              'arguments':{k:v for k,v in vars(a).items() if k not in ['resume','max_steps','epochs','checkpoint_steps']}}
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    step=epoch=entry_cursor=batch_cursor=0;best=math.inf;start=time.monotonic()
    if a.resume and (out/'last.pt').exists():
        old=torch.load(out/'last.pt',map_location=a.device,weights_only=True)
        if old['identity_sha256']!=identity_hash:raise ValueError('Resume source/input/config identity changed')
        model.load_state_dict(old['model_state']);optimizer.load_state_dict(old['optimizer_state'])
        step=old['step'];epoch=old['epoch'];entry_cursor=old['entry_cursor'];batch_cursor=old['batch_cursor'];best=old['best_validation_nll']
        torch.set_rng_state(old['rng_state'].cpu())
        if a.device.startswith('cuda') and old.get('cuda_rng_state') is not None:torch.cuda.set_rng_state_all([v.cpu() for v in old['cuda_rng_state']])
    elif (out/'last.pt').exists():
        raise ValueError('Existing run requires --resume or a new output directory')
    (out/'training_manifest.json').write_text(json.dumps(identity|{'identity_sha256':identity_hash,
        'software':{'torch':str(torch.__version__),'numpy':np.__version__},
        'parameters':sum(p.numel() for p in model.parameters()),'model_version':MODEL_VERSION},indent=2)+'\n')
    if a.max_steps and step>=a.max_steps:
        print(json.dumps({'event':'already_at_step_limit','step':step}));return
    def save(path,ep,ent,bat,metrics=None):
        atomic_save({'model_version':MODEL_VERSION,'config':asdict(config),'model_state':model.state_dict(),
            'optimizer_state':optimizer.state_dict(),'identity_sha256':identity_hash,'step':step,
            'epoch':ep,'entry_cursor':ent,'batch_cursor':bat,'best_validation_nll':best,
            'rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else None,
            'temperature':1.,'validation_metrics':metrics,'training_samples':train['samples'],
            'source_sha256':source,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},path)
    log=(out/'training.jsonl').open('a',buffering=1)
    if step==0:
        initial=evaluate(model,val,a.batch_size,a.device);log.write(json.dumps({'event':'initial_validation',**initial})+'\n')
    for ep in range(epoch,a.epochs):
        model.train();total=loss_sum=0
        for x,y,ent,bat,_ in batches(train,a.batch_size,ep,a.seed,True,entry_cursor if ep==epoch else 0,batch_cursor if ep==epoch else 0):
            batch=tensors(x,a.device);labels=torch.as_tensor(y,device=a.device)
            optimizer.zero_grad(set_to_none=True);result=model(batch,return_aux=True)
            main_loss=nn.functional.cross_entropy(result['logits'],labels)
            auxiliary=nn.functional.cross_entropy(result['read_logits'],labels)
            loss=main_loss+.15*auxiliary
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();step+=1
            loss_sum+=float(loss.detach())*len(y);total+=len(y)
            if step%a.checkpoint_steps==0:
                save(out/'last.pt',ep,ent,bat+1)
                progress={'event':'progress','epoch':ep,'step':step,'loss':loss_sum/total,'seconds':time.monotonic()-start}
                log.write(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
            if a.max_steps and step>=a.max_steps:
                save(out/'last.pt',ep,ent,bat+1)
                metrics=evaluate(model,val,a.batch_size,a.device)
                if metrics['nll']<best:best=metrics['nll'];save(out/'best.pt',ep,ent,bat+1,metrics)
                save(out/'last.pt',ep,ent,bat+1,metrics)
                log.write(json.dumps({'event':'step_limit','step':step,**metrics})+'\n');log.close();return
        entry_cursor=batch_cursor=0
        metrics=evaluate(model,val,a.batch_size,a.device)
        if metrics['nll']<best:best=metrics['nll'];save(out/'best.pt',ep+1,0,0,metrics)
        save(out/'last.pt',ep+1,0,0,metrics)
        # Retain every epoch for a prespecified independent full-truth validation
        # comparison; a reference-dominated NLL must not hide discovery failures.
        save(out/f'epoch-{ep+1:02d}.pt',ep+1,0,0,metrics)
        report={'event':'epoch_complete','epoch':ep+1,'step':step,'train_loss':loss_sum/max(1,total),**metrics,'seconds':time.monotonic()-start}
        log.write(json.dumps(report)+'\n');print(json.dumps(report),flush=True)
    log.close()
    (out/'TRAINING_COMPLETE.json').write_text(json.dumps({'step':step,'epochs':a.epochs,'seconds':time.monotonic()-start,
        'best_validation_nll':best,'best_checkpoint_sha256':sha(out/'best.pt'),'test_truth_used':False},indent=2)+'\n')

if __name__=='__main__':main()
