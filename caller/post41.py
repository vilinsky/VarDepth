"""Portable frozen VarDepth correction stages. No truth/teacher input exists."""
import argparse,json,pickle,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'postlib'))
import numpy as np,pysam
from threadpoolctl import threadpool_limits
import pipeline32 as p

def main():
 a=argparse.ArgumentParser()
 for name in ['stage','native','bam','reference','sample','chrom','out']:a.add_argument('--'+name,required=True)
 v=a.parse_args();out=Path(v.out);out.mkdir(parents=True,exist_ok=True);sample=v.sample
 with pysam.FastaFile(v.reference) as f:length=f.get_reference_length(v.chrom)
 p.R=out;p.PLAN={k:dict(samples=[sample] if k=='test' else [],chrom=v.chrom,intervals=[[0,length]]) for k in ['train','validation','test']}
 (out/'refs').mkdir(exist_ok=True)
 dest=out/'refs'/(v.chrom+'.fa')
 if not dest.exists():dest.symlink_to(Path(v.reference).resolve())
 if not Path(str(dest)+'.fai').exists():Path(str(dest)+'.fai').symlink_to(Path(v.reference+'.fai').resolve())
 source=out/'inputs'/sample;source.mkdir(parents=True,exist_ok=True)
 for suffix in ['', '.tbi']:
  target=source/('baseline.vcf.gz'+suffix)
  if not target.exists():target.symlink_to(Path(v.native+suffix).resolve())
 # Legacy inference-region mask, deliberately independent of benchmark truth.
 (source/'truth.bed').write_text(f'{v.chrom}\t0\t{length}\n')
 assert not (source/'truth.vcf.gz').exists()
 models=HERE/'models'
 with threadpool_limits(1):
  if v.stage=='prepare':
   d=p.prepare_stage(sample);w=p.load(models/'stage21.npz')
   z=np.concatenate([p.logits(d['x'][i:i+8192],w) for i in range(0,len(d['x']),8192)])
   q=np.concatenate([p.softmax(z[a:b]) for a,b in zip(d['offsets'][:-1],d['offsets'][1:])])
   with (models/'reliability17.pkl').open('rb') as f:m=pickle.load(f)
   reliability=m.predict_proba(d['x'][d['offsets'][:-1],:44])[:,1]
   import stage21 as old
   old.prepared=lambda s,stage:p.prepared(s);old.save=p.save
   stage21=out/'stage21'/sample/'calls.vcf.gz';old.write(sample,json.loads((HERE/'config/stage21.json').read_text()),d,q,reliability,stage21)
   d=p.components(sample,p.prepared(sample)/'tagged.vcf.gz',out/'raw_components'/sample)
   with (models/'component25.pkl').open('rb') as f:m=pickle.load(f)
   q=np.zeros((len(d['x']),4))
   for k in [False,True]:
    mask=d['indel']==k
    if mask.any():q[mask]=m[k].predict_proba(d['x'][mask])
   import component_inference32 as ci
   ci.CHROM=v.chrom;ci.REF=Path(v.reference);ci.baseline=lambda s:stage21;ci.save=p.save
   ci.write(sample,dict(confidence=.9,gain=.1,indel_weight=1.),d,q,p.baseline(sample))
   p.components(sample,p.baseline(sample),out/'residual'/sample,label=False,extra=True)
  elif v.stage=='reads':
   import read_features32 as read
   read.run(argparse.Namespace(chrom=v.chrom,bam=v.bam,reference=v.reference,keys=str(out/'residual'/sample/'keys.json'),out=str(out/'evidence-test'/sample),start=0,end=length))
  elif v.stage in ['predict','confidence']:
   d=p.load(out/'residual'/sample/'features.npz');d['x']=np.column_stack([d['x'],p.load(out/'evidence-test'/sample/'read_features.npz')['x']]).astype(np.float32)
   assert d['x'].shape[1]==139 and np.isfinite(d['x']).all()
   with (models/'read32.pkl').open('rb') as f:m=pickle.load(f)
   q=np.zeros((len(d['x']),4),np.float64)
   for k in [False,True]:
    mask=d['indel']==k
    if mask.any():q[mask]=m[k].predict_proba(d['x'][mask])
   calls=out/'candidates/catalogue70'/sample/'calls.vcf.gz'
   if v.stage=='predict':p.write_candidate(sample,json.loads((HERE/'config/residual32.json').read_text()),d,q,calls)
   else:
    import confidence33 as c
    c.ROOT=out;c.MODELS=models;c.data=lambda s:d;c.calls=lambda s,arm:calls;c.student_probabilities=lambda s,arm,data:q;c.save=p.save;c.sha=p.sha;c.load=p.load
    c.paths=lambda s:dict(root=out,chrom=v.chrom,ref=Path(v.reference),calls=calls,end=length)
    c.write(sample,'frozen')
  else:raise ValueError('Unknown frozen stage')
if __name__=='__main__':main()
