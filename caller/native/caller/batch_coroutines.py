"""Bounded ordered execution of independent likelihood coroutines."""
from collections import deque

def score_generators(generators,score,max_pairs=32768):
    if max_pairs<1:raise ValueError('max_pairs must be positive')
    result=[];pending=[];jobs=[]
    def advance(i,generator,value=None,first=False):
        try:
            request=next(generator) if first else generator.send(value)
            return (i,generator,request)
        except StopIteration as done:result[i]=done.value;return None
    def flush():
        nonlocal pending,jobs
        queue=deque(pending);pending=[];jobs=[]
        while queue:
            active=[];count=0
            while queue and (not active or count+len(queue[0][2])<=max_pairs):
                item=queue.popleft();active.append(item);count+=len(item[2])
            request_jobs=[job for i,generator,request in active for job in request]
            scored=score(request_jobs)
            if len(scored)!=len(request_jobs):raise ValueError('Likelihood scorer returned the wrong number of results')
            offset=0
            for i,generator,request in active:
                item=advance(i,generator,scored[offset:offset+len(request)]);offset+=len(request)
                if item is not None:queue.append(item)
    for i,generator in enumerate(generators):
        result.append(None);item=advance(i,generator,first=True)
        if item is None:continue
        if jobs and len(jobs)+len(item[2])>max_pairs:flush()
        pending.append(item);jobs.extend(item[2])
        if len(jobs)>=max_pairs:flush()
    flush();return result
