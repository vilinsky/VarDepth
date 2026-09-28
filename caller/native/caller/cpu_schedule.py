"""Bounded CPU scheduling from process affinity and inherited cgroup quotas."""
import math,os
from pathlib import Path

def available_cpus():
    count=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    try:
        entry=next(line.split(':',2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
        root=Path('/sys/fs/cgroup');path=root/entry.lstrip('/')
        for directory in [path,*path.parents]:
            if directory==root.parent:break
            if not directory.is_relative_to(root):continue
            file=directory/'cpu.max'
            if file.exists():
                quota,period=file.read_text().split()
                if quota!='max':count=min(count,max(1,math.floor(int(quota)/int(period))))
    except (OSError,ValueError,StopIteration):pass
    return max(1,count)

def allocate(budget,chunks,components,jobs=None,component_jobs=None,evidence_jobs=None,neural_threads=1):
    if budget<1 or chunks<1 or components<1:raise ValueError('Positive CPU budget and task counts required')
    for value in (jobs,component_jobs,evidence_jobs):
        if value is not None and value<1:raise ValueError('Worker counts must be positive')
    if neural_threads<1:raise ValueError('Neural thread count must be positive')
    minimum=max(neural_threads,evidence_jobs or 1)
    requested_components=min(components,component_jobs or 1)
    # Prefer independent genomic work: it stays CPU-efficient and avoids
    # serial HMMs leaving cores idle inside a single over-threaded window.
    j=min(chunks,jobs if jobs is not None else max(1,budget//requested_components//minimum))
    c=min(components,component_jobs if component_jobs is not None else max(1,budget//j//minimum))
    local=max(1,budget//j//c)
    e=evidence_jobs if evidence_jobs is not None else max(1,local-1)
    if j*c*max(e,neural_threads)>budget:raise ValueError('Requested process concurrency exceeds the available CPU budget')
    return dict(budget=budget,jobs=j,component_jobs=c,evidence_jobs=e,component_budget=local,
                peak_evidence_workers=j*c*e,peak_neural_threads=j*c*neural_threads,
                scope='Independent windows first; bounded logical CPU workers; physical/SMT scaling measured separately')
