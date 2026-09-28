import collections,json,math,time
from pathlib import Path
import numpy as np,pysam
from core import decision,pairs

def write(sample,config,d,q,reliability,out):
    begin=time.monotonic();cpu=time.process_time();out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);indices={str(v):i for i,v in enumerate(d['ids'])};stats=collections.Counter()
    with pysam.VariantFile(prepared(sample,config['stage'])/'tagged.vcf.gz') as source:
        h=source.header.copy()
        for name,n,t,desc in [('R21P',1,'Float','Experimental conditional genotype mass'),('R21CC',1,'Float','Experimental component consensus, not calibrated error probability'),('R21OGT',1,'String','Original unphased genotype')]:
            if name not in h.formats:h.formats.add(name,n,t,desc)
        h.add_line('##vardepth_v21=research; GP is conditional on retained genotype hypotheses; not clinical calibration')
        with pysam.VariantFile(out,'wz',header=h) as dest:
            for rec in source:
                rec.translate(h);stats['records']+=1;i=indices.get(rec.id)
                if i is not None:
                    c=rec.samples[sample];old=tuple(c['GT']);lo,hi=d['offsets'][i:i+2];a,b=d['cost_offsets'][i:i+2];cost=d['cost'][a:b].reshape(hi-lo,hi-lo);choices=d['choices'][lo:hi]
                    chosen,abstain,detail=decision(q[lo:hi],cost,choices,int(d['current'][i]),config,float(reliability[i]),str(d['kinds'][i]));order=pairs(len(rec.alleles));newgt=order[chosen]
                    if any(abs(len(rec.alleles[k])-len(rec.ref))>50 for k in newgt):newgt=old;abstain=bool(any(old) and reliability[i]<{'SNV':.6,'INDEL':.3}[str(d['kinds'][i])]);stats['out_of_range_change_prevented']+=1
                    final=(None,None) if abstain and any(newgt) else newgt
                    c['R21P']=detail['p'];c['R21CC']=detail['component_consensus'];c['R21OGT']='/'.join(map(str,old))
                    gp=np.zeros(len(order));gp[choices]=q[lo:hi];c['GP']=tuple(map(float,gp));c['GQ']=min(99,int(round(-10*math.log10(max(1e-10,1-float(gp[chosen]))))))
                    dosage=np.zeros(len(rec.alts))
                    for p,(a,b) in zip(gp,order):
                        if a:dosage[a-1]+=p
                        if b:dosage[b-1]+=p
                    c['DS']=tuple(map(float,dosage));c['GT']=final
                    if final!=old:
                        stats['changed']+=1;stats['no_call' if None in final else 'GT_switch']+=1;c.phased=False
                        for field in ['PS','VPE','VPSRC']:
                            if field in c:c[field]=None
                    rec.qual=min(99.,-10*math.log10(max(1e-10,float(gp[0]))));rec.filter.clear();rec.filter.add('PASS' if c['GQ']>=20 and None not in final else 'LowQual')
                dest.write(rec)
    pysam.tabix_index(str(out),preset='vcf',force=True);result=dict(success=True,config=config,stats=dict(stats),cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-begin,conditional_GP=True);save(str(out)+'.json',result);return result
