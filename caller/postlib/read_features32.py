"""Truth-blind CIGAR/molecule evidence. No truth, panel or comparator access.

Quality-weighted mate consensus counts each library/read-name once. Reference
support for an indel requires aligned sequence spanning both event flanks;
other local indels count as ambiguous, never as reference support.
"""
import argparse,bisect,collections,hashlib,json,math,time
from pathlib import Path
import numpy as np
import pysam
from candidate_normalizer import normalize_biallelic

NAMES=['log_dp','log_ref','log_alt','log_other','ref_fraction','alt_fraction',
       'other_fraction','alt_strand_min_fraction','ref_strand_min_fraction',
       'alt_distinct_start_fraction','mate_disagreement_fraction',
       'ref_mq','alt_mq','other_mq','ref_bq','alt_bq','other_bq',
       'ref_end_distance','alt_end_distance','other_end_distance',
       'ref_nm_fraction','alt_nm_fraction','other_nm_fraction',
       'ref_clip_fraction','alt_clip_fraction','other_clip_fraction',
       'ref_proper_fraction','alt_proper_fraction','other_proper_fraction',
       'log_weak_mq','log_weak_bq','read_dosage0','read_dosage1','read_dosage2',
       'log_length','insertion','repeat1','repeat2','repeat3','repeat4']

def valid(r):
 return not(r.is_unmapped or r.is_secondary or r.is_supplementary or r.is_duplicate or r.is_qcfail) and r.query_sequence is not None and r.query_qualities is not None

def parse(r,seq,cache):
 p=r.reference_start;q=0;segments=[];events=[];raw=[];clip=0
 for op,n in r.cigartuples or []:
  if op in (0,7,8):segments.append((p,p+n,q));p+=n;q+=n
  elif op in (1,2):
   start=p-1;raw.append((start,p+(n if op==2 else 0)))
   if start>=0 and 1<=n<=100 and q>=3 and q+(n if op==1 else 0)+3<=len(r.query_sequence):
    ref=seq[start:p+(n if op==2 else 0)];alt=seq[start:p]+(r.query_sequence[q:q+n] if op==1 else '')
    qual=min(r.query_qualities[q-3:q+(n if op==1 else 0)+3]);key=(start,ref,alt)
    if not (set(ref+alt)-set('ACGT')):
     if key not in cache:
      try:cache[key]=normalize_biallelic(*key,seq)
      except ValueError:cache[key]=None
     if cache[key] is not None:events.append((cache[key],q,qual))
   if op==1:q+=n
   else:p+=n
  elif op==3:p+=n
  elif op==4:q+=n;clip+=n
 return segments,events,raw,clip

def molecule(r,rg):return (rg.get(r.get_tag('RG'),'unknown') if r.has_tag('RG') else 'unknown',r.query_name)

def repeat(seq,period):
 best=run=0
 for i in range(period,len(seq)):
  run=run+1 if seq[i]==seq[i-period] else 0;best=max(best,run)
 return best+period

def summarize(key,mols,weak,sequence):
 rows=list(mols.values());dp=len(rows);groups=[[r for r in rows if r[0]==k] for k in (0,1,2)];counts=[len(g) for g in groups];ref,alt,other=counts
 feat=[math.log1p(dp),*[math.log1p(n) for n in counts],*[n/max(1,dp) for n in counts]]
 for g in [groups[1],groups[0]]:feat.append(min(sum(x[3]==s for x in g) for s in [0,1])/max(1,len(g)))
 feat.extend([len({r[4] for r in groups[1]})/max(1,alt),sum(r[-1] for r in rows)/max(1,dp)])
 for ix in [1,2,5,6,7,8]:feat.extend([sum(r[ix] for r in g)/len(g) if g else -1. for g in groups])
 feat.extend([math.log1p(weak[0]),math.log1p(weak[1])]);ll=np.asarray([ref*math.log(.99)+alt*math.log(.01),(ref+alt)*math.log(.5),ref*math.log(.01)+alt*math.log(.99)]);q=np.exp(ll-ll.max());q/=q.sum();feat.extend(q.tolist());p,r,a=key;context=sequence[max(0,p-30):p+len(r)+30];feat.extend([math.log1p(abs(len(a)-len(r))),float(len(a)>len(r)),*[repeat(context,v) for v in (1,2,3,4)]])
 assert len(feat)==len(NAMES)
 return feat,dict(molecules=dp,ref_exact=ref,alt_exact=alt,other=other,alt_strands=[sum(r[3]==s for r in groups[1]) for s in [0,1]])

def update(store,mol,row):
 old=store.get(mol)
 if old is None:store[mol]=row;return
 discord=old[-1] or old[0]!=row[0]
 # Agreeing mates count once; conflicting mates use the higher quality read.
 chosen=row if (row[1],row[2])>(old[1],old[2]) else old
 store[mol]=chosen[:-1]+(int(discord),)

def run(args):
 t=time.monotonic();cpu=time.process_time();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);keys=[tuple(k) for k in json.loads(Path(args.keys).read_text())];known=set(keys);seq=pysam.FastaFile(args.reference).fetch(args.chrom).upper();cache={};discovery=collections.defaultdict(dict);stats=collections.Counter()
 with pysam.AlignmentFile(args.bam,'rb') as bam:
  rg={x['ID']:x.get('LB',x['ID']) for x in bam.header.to_dict().get('RG',[])}
  for read in bam.fetch(args.chrom,max(0,args.start-1000),args.end+1000):
   if not valid(read) or not 20<=read.mapping_quality<255:continue
   _,events,_,_=parse(read,seq,cache)
   for k,qp,bq in events:
    if args.start<=k[0]<args.end and bq>=20 and k not in known:discovery[k][molecule(read,rg)]=read.is_reverse
  novel={k for k,v in discovery.items() if len(v)>=3};allkeys=sorted(known|novel);positions=[k[0] for k in allkeys];result={};notes={}
  # Bounded memory, with 1kb overlap to retain repeat-shifted CIGAR support.
  for start in range(args.start,args.end,250000):
   end=min(args.end,start+250000);lo=bisect.bisect_left(positions,start);hi=bisect.bisect_left(positions,end);chunk=allkeys[lo:hi];cp=positions[lo:hi];stores={k:{} for k in chunk};weak=collections.defaultdict(lambda:[0,0])
   for read in bam.fetch(args.chrom,max(0,start-1000),end+1000):
    if not valid(read):continue
    segments,events,raw,clip=parse(read,seq,cache);mol=molecule(read,rg);mq=read.mapping_quality;nm=float(read.get_tag('NM'))/max(1,read.query_alignment_length) if read.has_tag('NM') else -1.;obs={}
    for a,b,q0 in segments:
     i=bisect.bisect_left(cp,a);j=bisect.bisect_left(cp,b)
     for key in chunk[i:j]:
      p,ref,alt=key;qp=q0+p-a
      if len(ref)==len(alt)==1:
       base=read.query_sequence[qp];state=0 if base==ref else 1 if base==alt else 2;bq=read.query_qualities[qp]
      else:
       if a>p-3 or b<p+len(ref)+3:continue
       state=2 if any(max(p-3,x)<min(p+len(ref)+3,y+1) for x,y in raw) else 0;bq=min(read.query_qualities[qp-3:qp+len(ref)+3])
      obs[key]=(state,qp,bq)
    for key,qp,bq in events:
     if key in stores:obs[key]=(1,qp,bq)
    for key,(state,qp,bq) in obs.items():
     if not 20<=mq<255:weak[key][0]+=1;continue
     if bq<15:weak[key][1]+=1;continue
     row=(state,mq,bq,int(read.is_reverse),read.reference_start,min(qp,len(read.query_sequence)-qp-1),nm,clip/max(1,len(read.query_sequence)),int(read.is_proper_pair),0);update(stores[key],mol,row)
   for key in chunk:result[key],notes[key]=summarize(key,stores[key],weak[key],seq)
   stats['chunks']+=1
  # Boundary keys may lie outside the registered interval in old decomposed blocks.
  for key in known-set(result):result[key],notes[key]=summarize(key,{},[0,0],seq);stats['outside_interval_keys']+=1
 np.savez_compressed(out/'read_features.npz',x=np.asarray([result[k] for k in keys],np.float32))
 rows=[dict(key=k,features=result[k],origins=['cigar'],**notes[k]) for k in sorted(novel) if k in result]
 (out/'novel.json').write_text(json.dumps(rows));meta=dict(success=True,features=NAMES,known_rows=len(keys),novel_candidates=len(rows),stats=dict(stats),cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-t,truth_used=False,interval=[args.start,args.end],keys_sha256=hashlib.sha256(Path(args.keys).read_bytes()).hexdigest(),bam=args.bam)
 (out/'COMPLETE.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--chrom',required=True)
 for k in ['bam','reference','keys','out']:p.add_argument('--'+k,required=True)
 for k in ['start','end']:p.add_argument('--'+k,type=int,required=True)
 run(p.parse_args())
