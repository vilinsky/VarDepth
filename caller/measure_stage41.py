"""Portable component accounting; aggregate sampled RSS is not physical RAM."""
import hashlib,json,os,resource,subprocess,time
from pathlib import Path
import psutil
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def measure(command,directory,env=None):
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
 (directory/'COMMAND.json').write_text(json.dumps(command,indent=2)+'\n')
 start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);peak=0;samples=0;last=start;gap=0
 with (directory/'run.log').open('wb') as log:
  process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
  while True:
   now=time.monotonic();gap=max(gap,now-last);last=now;rss=0
   try:
    p=psutil.Process(process.pid)
    for q in [p]+p.children(recursive=True):
     try:rss+=q.memory_info().rss
     except (psutil.NoSuchProcess,psutil.AccessDenied):pass
   except psutil.NoSuchProcess:pass
   peak=max(peak,rss);samples+=1
   if process.poll() is not None:break
   time.sleep(.1)
 elapsed=time.monotonic()-start;after=resource.getrusage(resource.RUSAGE_CHILDREN)
 result=dict(success=process.returncode==0,returncode=process.returncode,wall_seconds=elapsed,
  cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
  sampled_aggregate_RSS_peak_bytes=peak,memory_samples=samples,max_sample_gap_seconds=gap,
  cpu_definition='RUSAGE_CHILDREN delta for waited component process tree; sequential measurement wrapper',
  memory_definition='100ms sampled sum of process-tree RSS; shared pages may be counted more than once; lower bound on peak aggregate RSS, not cgroup physical memory')
 result['cpu_hours']=result['cpu_seconds']/3600
 (directory/'RESOURCES.json').write_text(json.dumps(result,indent=2)+'\n')
 if process.returncode:raise subprocess.CalledProcessError(process.returncode,command)
 return result
