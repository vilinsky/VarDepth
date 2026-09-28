import collections,hashlib,json,bisect,math,time
from pathlib import Path
import numpy as np,pysam
from v25_core import eligible,candidate_choices,feature_rows,gt_index,event_data,pairs,allele_events,short_event
from learning import kind,probability
from core import logits,softmax
from set_truth_labeler import ConfidentBed

def configure(sample):
 global CHROM,REF,REF_SDF
 CHROM=PLAN[role(sample)]['chrom'];REF=R/'refs'/(CHROM+'.fa');REF_SDF=R/'refs'/(CHROM+'.sdf')
 assert PLAN[role(sample)]['intervals'][0][1]<=pysam.FastaFile(REF).get_reference_length(CHROM)

def save(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')

def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()

def load(p):
 with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}

def role(sample):return next(k for k in ['train','validation','test'] if sample in PLAN[k]['samples'])

def prepared(sample):return R/'prepared'/sample

def baseline(sample):return R/'baseline'/sample/'calls.vcf.gz'

def prepare_stage(sample,src=None,intervals=None):
 configure(sample)
 out=prepared(sample);out.mkdir(parents=True,exist_ok=False);src=Path(src) if src else R/'inputs'/sample;regions=intervals or PLAN[role(sample)]['intervals'];start=time.monotonic();cpu=time.process_time()
 mask=ConfidentBed(src/'truth.bed');spans=sorted((max(s,a),min(e,b)) for s,e in mask.intervals[CHROM] for a,b in regions if min(e,b)>max(s,a));(out/'truth.bed').write_text(''.join(f'{CHROM}\t{a}\t{b}\n' for a,b in spans));starts=[a for a,b in spans];seq=pysam.FastaFile(REF).fetch(CHROM).upper();g=collections.defaultdict(list);offsets=[0];coff=[0];stats=collections.Counter()
 with pysam.VariantFile(src/'baseline.vcf.gz') as f,pysam.VariantFile(out/'tagged.vcf.gz','wz',header=f.header) as dst:
  for a,b in regions:
   for rec in f.fetch(CHROM,a,b):
    if not a<=rec.start<b:continue
    stats['records']+=1;rec.id='R24_'+str(stats['records'])
    if set(rec.ref.upper())-set('ACGTN'):
     i=bisect.bisect_left(starts,rec.stop)-1
     if i>=0 and spans[i][1]>rec.start:raise ValueError('Invalid REF in mask')
     stats['outside_mask_invalid_REF_removed']+=1;continue
    dst.write(rec)
    if not eligible(rec,sample) or len(rec.ref)>2000:continue
    ch=candidate_choices(rec,sample,True);x,cost=feature_rows(rec,sample,seq,ch)
    for k,v in [('x',x),('choices',ch),('ids',rec.id),('current',gt_index(rec.samples[sample]['GT'])),('kinds',kind(rec,sample)),('cost',cost.ravel())]:g[k].append(v)
    offsets.append(offsets[-1]+len(ch));coff.append(coff[-1]+cost.size);stats['eligible']+=1
 d={k:np.concatenate(v) if k in ['x','choices','cost'] else np.asarray(v) for k,v in g.items()};d.update(offsets=np.asarray(offsets),cost_offsets=np.asarray(coff));np.savez_compressed(out/'features.npz',**d);pysam.tabix_index(str(out/'tagged.vcf.gz'),preset='vcf',force=True)
 save(out/'REGISTERED.json',dict(sample=sample,regions=regions,source={k:str(src/v) for k,v in [('baseline','baseline.vcf.gz'),('truth','truth.vcf.gz'),('bed','truth.bed')]},truth_used=False))
 save(out/'COMPLETE.json',dict(stats=dict(stats),cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start));return d

def components(sample,vcf,out,label=False,extra=False):
 configure(sample)
 out=Path(out);out.mkdir(parents=True,exist_ok=False);seq=pysam.FastaFile(REF).fetch(CHROM).upper();meta=json.loads((prepared(sample)/'REGISTERED.json').read_text());labeler=None
 if label:
  assert role(sample)=='train'
  from chromosome_labels import ComponentTruth
  labeler=ComponentTruth(meta['source']['truth'],prepared(sample)/'truth.bed',seq,REF,chrom=CHROM)
 xs=[];ys=[];ids=[];keys=[];offsets=[0];doses=[];doff=[0];choices=[];coff=[0];stats=collections.Counter();cpu=time.process_time();begin=time.monotonic()
 with pysam.VariantFile(vcf) as source:
  for rec in source:
   if not eligible(rec,sample) or len(rec.ref)>2000:continue
   try:z=event_data(rec,sample,seq)
   except ValueError as error:
    if str(error) not in ['noncanonical_normalization','normalization_ref_mismatch','normalization_limit','Unanchored chromosome-start event']:raise
    stats['unusable_projection_'+str(error)]+=1;continue
   if z is None:continue
   k,d,x=z
   if extra:
    rows=[];c=rec.samples[sample];current=gt_index(c['GT']);ng=len(pairs(len(rec.alleles)))
    for j,(p,r,a) in enumerate(k):
     dist=np.asarray([abs(p-v[0]) for v in k]);other=np.arange(len(k))!=j;ki=np.asarray([len(v[1])!=len(v[2]) for v in k]);row=[math.log1p(min(dist[other],default=1000))]
     for radius in [10,25,50]:row.extend([float(np.sum(other&(dist<=radius)&(~ki))),float(np.sum(other&(dist<=radius)&ki))])
     for field in ['R21P','R21CC','R24MC','R24GAIN']:
      v=c.get(field);row.append(float(v) if v is not None and math.isfinite(v) else -1.)
     row.extend([float(c.get('R24OGT') is not None),float(c.get('R21OGT') is not None and c.get('R21OGT')!='/'.join(map(str,c['GT'])))])
     for field in ['GP','NGP','BGP']:
      q=probability(c,field,ng);row.append(float(d[int(q.argmax()),j]!=d[current,j]) if q is not None else -1.)
     rows.append(row)
    x=np.column_stack([x,np.asarray(rows,np.float32)])
   ch=candidate_choices(rec,sample,True);ids.append(rec.id);keys.extend(k);xs.append(x);doses.append(d.ravel());doff.append(doff[-1]+d.size);choices.extend(ch);coff.append(len(choices));offsets.append(offsets[-1]+len(k));stats['blocks']+=1;stats['components']+=len(k)
   if label:
    y=labeler.record_labels(rec,k,d);ys.extend(y);stats.update({'label_'+str(a):y.count(a) for a in [-1,0,1,2]})
 d=dict(x=np.concatenate(xs),y=np.asarray(ys,np.int8),ids=np.asarray(ids),offsets=np.asarray(offsets),dose=np.concatenate(doses),dose_offsets=np.asarray(doff),choices=np.asarray(choices),choice_offsets=np.asarray(coff),indel=np.asarray([len(r)!=len(a) for p,r,a in keys]))
 np.savez_compressed(out/'features.npz',**d);save(out/'keys.json',keys);save(out/'COMPLETE.json',dict(sample=sample,stats=dict(stats),training=label,truth_used=label,features=list(d['x'].shape),label_sources=dict(labeler.stats) if labeler else {},cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-begin));return d

def write_candidate(sample,config,d,q,out):
 configure(sample)
 out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);seq=pysam.FastaFile(REF).fetch(CHROM).upper();index={s:i for i,s in enumerate(d['ids'])};stats=collections.Counter();changes=[];start=time.monotonic();cpu=time.process_time()
 with pysam.VariantFile(baseline(sample)) as f:
  h=f.header.copy()
  for name,typ,desc in [('R30OGT','String','Original genotype before experimental residual correction'),('R30MC','Float','Minimum changed dosage confidence, uncalibrated'),('R30GAIN','Float','Expected component error reduction')]:h.formats.add(name,1,typ,desc)
  with pysam.VariantFile(out,'wz',header=h) as dest:
   for rec in f:
    rec.translate(h);c=rec.samples[sample];gt=c.get('GT');i=index.get(rec.id);stats['records']+=1
    if i is None or not gt or None in gt:dest.write(rec);continue
    a,b=d['offsets'][i:i+2];da,db=d['dose_offsets'][i:i+2];ca,cb=d['choice_offsets'][i:i+2];dose=d['dose'][da:db].reshape(-1,b-a);current=gt_index(gt);choices=np.unique(np.r_[d['choices'][ca:cb],current]);probs=q[a:b];loss=1-probs[np.arange(b-a)[None,:],dose[choices]];risk=loss.sum(1);best=int(np.argmin(risk));selected=int(choices[best]);changed=dose[selected]!=dose[current];oldpos=int(np.flatnonzero(choices==current)[0]);gain=float((risk[oldpos]-risk[best])/max(1,changed.sum()));threshold=np.where(d['indel'][a:b],config['indel_confidence'],config['snv_confidence']);confidence=probs[np.arange(b-a),dose[selected]]
    if changed.any() and gain>=config['min_gain'] and np.all(confidence[changed]>=threshold[changed]):
     order=pairs(len(rec.alleles));events=allele_events(rec,seq);outside={k for k in set().union(*events) if not short_event(k)}
     if any(sum(k in events[g] for g in gt)!=sum(k in events[g] for g in order[selected]) for k in outside):stats['out_of_scope_preserved']+=1;dest.write(rec);continue
     c['R30OGT']='/'.join(map(str,gt));c['GT']=order[selected];c.phased=False;c['R30MC']=float(confidence[changed].min());c['R30GAIN']=gain
     for field in ['GP','DS']:
      if field in c:c[field]=tuple([None]*(len(order) if field=='GP' else len(rec.alts)))
     for field in ['PS','VPE','VPSRC','GQ']:
      if field in c:c[field]=None
     rec.qual=None;rec.filter.clear();rec.filter.add('LowQual');stats['changed']+=1;changes.append(dict(pos1=rec.pos,id=rec.id,old_GT=gt,new_GT=order[selected],gain=gain,confidence=float(confidence[changed].min())))
    dest.write(rec)
 pysam.tabix_index(str(out),preset='vcf',force=True);save(str(out)+'.changes.json',changes);save(str(out)+'.json',dict(config=config,stats=dict(stats),truth_used=False,cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start));return out
