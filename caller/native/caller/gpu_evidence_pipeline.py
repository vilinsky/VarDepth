"""Bounded CPU evidence actors with GPU likelihoods only in the parent process.

Live generators and pysam objects stay in their owning worker. Only immutable
read/quality/haplotype requests and likelihood arrays cross process boundaries.
"""
import itertools
import multiprocessing as mp
import queue
import traceback
import numpy as np


def _tag(generator, anchored):
    try:
        request=next(generator)
        while True:request=generator.send((yield anchored,request))
    except StopIteration as done:return done.value


def full_steps(task):
    import evidence_workers as w
    from partial_anchor import _refine_steps
    from whole_read import _inputs,_finish
    row,original,depths=task;row=w.proposal((row,original));fetch=w.fetch_for(row['chrom'])
    updated,detail=yield from _tag(_refine_steps(row,original,w._BAMS,fetch,w._LIB),True)
    eligible=max(depths.values())>=20 and (len(updated['alleles'][0])>=12 or len(updated['alleles'])>=4)
    whole=None
    if eligible:
        extended,summary=w.graph((updated,depths))
        if summary['added']:
            extended=dict(extended,samples={name:dict(s,dp=s['original_spanning_dp']) for name,s in updated['samples'].items()})
            updated,detail=yield from _tag(_refine_steps(extended,original,w._BAMS,fetch,w._LIB),True)
        else:updated=extended
        samples={}
        for sample,inputs in w._BAMS.items():
            context=_inputs(updated,inputs,fetch);retained=context['retained'];haps=context['haps']
            requests=[(r[2],r[3],h) for r in retained for h in haps]
            ll=(yield False,requests) if requests else np.empty(0,np.float64)
            samples[sample]=_finish(updated,context,np.asarray(ll).reshape(len(retained),len(haps)))
        whole=dict(updated,samples=samples,whole_read_expert=True)
    return updated,detail,whole


def _actor(commands,responses,arguments,reference,factory):
    import os
    # This requests suspended likelihood jobs; this module never instantiates
    # CUDA in workers. The parent chooses the actual scorer.
    os.environ['VARDEPTH_PAIRHMM_BACKEND']='cuda'
    sessions={};identity=None
    try:
        if arguments is not None:
            from evidence_workers import initialize
            initialize(arguments,reference)
        while True:
            command=commands.get()
            if command is None:return
            kind,identity,payload=command
            if kind=='start':
                if identity in sessions:raise ValueError('Duplicate active evidence task')
                generator=factory(payload);sessions[identity]=generator;first=True
            elif kind=='resume':generator=sessions[identity];first=False
            else:raise ValueError('Unknown evidence actor command')
            try:
                anchored,requests=next(generator) if first else generator.send(payload)
                responses.put(('request',identity,bool(anchored),requests))
            except StopIteration as done:
                del sessions[identity];responses.put(('done',identity,done.value))
    except BaseException:
        responses.put(('error',identity,traceback.format_exc()))
    finally:
        for generator in sessions.values():generator.close()


def run_tasks(tasks,arguments,reference,*,workers=4,slots_per_worker=8,max_pairs=32768,scorer=None,factory=full_steps):
    if workers<1 or slots_per_worker<1 or max_pairs<1:raise ValueError('Invalid evidence pipeline bounds')
    if scorer is None:
        from accelerator import score_pairs,backend
        choice=backend()
        scorer=lambda requests,anchored:score_pairs(requests,anchored=anchored,selection=choice)
    tasks=iter(tasks)
    try:first=next(tasks)
    except StopIteration:return []
    tasks=itertools.chain([first],tasks);context=mp.get_context('spawn');window=workers*slots_per_worker
    responses=context.Queue(maxsize=2*window);commands=[context.Queue(maxsize=2*slots_per_worker) for _ in range(workers)]
    processes=[context.Process(target=_actor,args=(commands[i],responses,arguments,reference,factory)) for i in range(workers)]
    owners={};loads=[0]*workers;results=[];pending=[];inflight=0;exhausted=False;success=False
    def fill():
        nonlocal exhausted,inflight
        while not exhausted and len(owners)<window:
            try:task=next(tasks)
            except StopIteration:exhausted=True;break
            worker=min(range(workers),key=lambda i:loads[i]);identity=len(results);results.append(None)
            owners[identity]=worker;loads[worker]+=1;commands[worker].put(('start',identity,task));inflight+=1
    try:
        for process in processes:process.start()
        fill()
        while owners:
            if pending and (inflight==0 or sum(len(x[2]) for x in pending)>=max_pairs):
                mode=pending[0][1];selected=[];rest=[];count=0
                for item in pending:
                    if item[1]==mode and (not selected or count+len(item[2])<=max_pairs):selected.append(item);count+=len(item[2])
                    else:rest.append(item)
                pending=rest;requests=[p for _,_,jobs in selected for p in jobs];values=np.asarray(scorer(requests,mode))
                if values.ndim!=1 or len(values)!=count:raise ValueError('Likelihood scorer returned the wrong shape')
                offset=0
                for identity,anchored,jobs in selected:
                    commands[owners[identity]].put(('resume',identity,values[offset:offset+len(jobs)]));offset+=len(jobs);inflight+=1
                continue
            try:message=responses.get(timeout=2)
            except queue.Empty:
                if any(not p.is_alive() for p in processes):raise RuntimeError('Evidence worker exited before completing its tasks')
                continue
            kind,identity,*payload=message
            if kind=='error':raise RuntimeError('Evidence worker failed:\n'+payload[0])
            if identity not in owners:raise RuntimeError('Unowned evidence response')
            inflight-=1
            if kind=='request':pending.append((identity,*payload))
            elif kind=='done':
                results[identity]=payload[0];loads[owners.pop(identity)]-=1;fill()
            else:raise RuntimeError('Unknown evidence response')
        if pending or inflight:raise RuntimeError('Evidence accounting mismatch')
        success=True;return results
    finally:
        if success:
            for q in commands:q.put(None)
            for p in processes:p.join(timeout=10)
        for p in processes:
            if p.is_alive():p.terminate();p.join(timeout=2)
        for q in [*commands,responses]:
            if not success:q.cancel_join_thread()
            q.close()
            if success:q.join_thread()
