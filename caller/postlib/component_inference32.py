import collections,json,time
import numpy as np,pysam
from v25_core import choose,pairs,gt_index,allele_events,short_event

def write(sample,cfg,data,q,out):
 out.parent.mkdir(parents=True,exist_ok=True);indices={s:i for i,s in enumerate(data['ids'])};stats=collections.Counter();start=time.process_time();sequence=pysam.FastaFile(REF).fetch(CHROM).upper();diagnostics=[]
 with pysam.VariantFile(baseline(sample)) as source:
  h=source.header.copy();h.formats.add('R24MC',1,'Float','Minimum learned dosage probability among changed components; research score, not genotype calibration');h.formats.add('R24GAIN',1,'Float','Expected component-error reduction per changed component');h.formats.add('R24OGT',1,'String','Genotype before component decision')
  with pysam.VariantFile(out,'wz',header=h) as dst:
   for rec in source:
    rec.translate(h);stats['records']+=1;i=indices.get(rec.id);c=rec.samples[sample];gt=c.get('GT')
    if i is None or not gt or None in gt:dst.write(rec);continue
    a,b=data['offsets'][i:i+2];da,db=data['dose_offsets'][i:i+2];ca,cb=data['choice_offsets'][i:i+2];dose=data['dose'][da:db].reshape(-1,b-a);current=gt_index(gt);choices=np.unique(np.r_[data['choices'][ca:cb],current]);order=pairs(len(rec.alleles))
    selected,detail=choose(dose,q[a:b],choices,current,cfg['confidence'],cfg['gain'],data['indel'][a:b],cfg['indel_weight'])
    if selected!=current:
     events=allele_events(rec,sequence);outside={k for k in set().union(*events) if not short_event(k)}
     if any(sum(k in events[g] for g in gt)!=sum(k in events[g] for g in order[selected]) for k in outside):stats['out_of_scope_preserved']+=1;dst.write(rec);continue
     c['R24OGT']='/'.join(map(str,gt));c['GT']=order[selected];c.phased=False;c['R24MC']=detail['min_component_confidence'];c['R24GAIN']=detail['gain']
     # Marginal component predictions do not define a calibrated joint GP.
     for field in ['GP','DS']:
      if field in c:c[field]=tuple([None]*(len(order) if field=='GP' else len(rec.alts)))
     for field in ['PS','VPE','VPSRC','GQ']:
      if field in c:c[field]=None
     rec.qual=None;rec.filter.clear();rec.filter.add('LowQual');stats['changed']+=1;diagnostics.append(dict(id=rec.id,pos1=rec.pos,old_GT=gt,new_GT=order[selected],**detail))
    dst.write(rec)
 pysam.tabix_index(str(out),preset='vcf',force=True);save(str(out)+'.changes.json',diagnostics);save(str(out)+'.json',dict(success=True,config=cfg,stats=dict(stats),cpu_seconds=time.process_time()-start,changed_joint_GP_missing=True,truth_used=False))
