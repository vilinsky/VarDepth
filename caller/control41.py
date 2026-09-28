"""Measured coverage routing and immutable-input identities, without truth."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'postlib'))
import pysam
from apply_guard36 import make_profile
from repair36 import repair,save,sha
def main():
 p=argparse.ArgumentParser()
 for n in ['stage','native','reference','sample','chrom','out']:p.add_argument('--'+n,required=True)
 p.add_argument('--current');p.add_argument('--bam');p.add_argument('--panel');p.add_argument('--genetic-map');a=p.parse_args();out=Path(a.out)
 guard=json.loads((HERE/'config/GUARD.json').read_text())
 if a.stage=='profile':
  with pysam.FastaFile(a.reference) as f:length=f.get_reference_length(a.chrom)
  save(out/'PROFILE.json',make_profile(a.native,a.sample,a.chrom,length,guard))
 elif a.stage=='select':
  profile=json.loads((out/'PROFILE.json').read_text())
  repair(dict(id=a.sample+'-'+a.chrom,sample=a.sample,chrom=a.chrom,native=a.native,current=a.current),profile,guard,out/'guard',policies=('coverage-domain-guard',))
 elif a.stage=='identities':
  rows={}
  for name in [a.bam,a.reference,a.panel,a.genetic_map]:
   path=Path(name).resolve();before=path.stat();digest=sha(path);after=path.stat()
   assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Input changed while hashing'
   rows[str(path)]=dict(bytes=after.st_size,mtime_ns=after.st_mtime_ns,sha256=digest)
  save(out/'PHASE_INPUT_IDENTITIES.json',rows)
 else:raise ValueError('Unknown stage')
if __name__=='__main__':main()
