"""Discriminative mixture of frozen neural, LD scaffold and anchored experts.

Arithmetic mixtures avoid multiplying two posteriors derived from the same reads.
New local alleles may exceed the neural head's original 12-allele capacity; the
old GP is embedded without changing existing allele numbering. No truth input.
"""
import argparse,hashlib,json,math,pathlib,time
import numpy as np
import pysam

DEFAULT=dict(scaffold_weight=.9,scaffold_depth_scale=4.,indel_base_weight=0.,
             indel_base_midpoint=75.,indel_base_width=15.,anchored_weight=.8,anchored_min_minor_fraction=.25)

def order(n):return [(a,b) for b in range(n) for a in range(b+1)]
def project(values,n,fill=0.):
    out=np.full(n*(n+1)//2,fill,dtype=float);out[:len(values)]=values;return out
def normalized(x):
    x=np.array(x,dtype=float)
    if not np.isfinite(x).all() or np.any(x<0) or x.sum()<=0:raise ValueError('Invalid expert probability')
    return x/x.sum()

def run(args):
    begun=time.monotonic();config=dict(DEFAULT)
    if args.config:config.update(json.loads(pathlib.Path(args.config).read_text()))
    for k in ['scaffold_weight','indel_base_weight','anchored_weight']:
        if not 0<=config[k]<=1:raise ValueError('Mixture weights must be 0..1')
    if min(config['scaffold_depth_scale'],config['indel_base_width'])<=0:raise ValueError('Invalid depth scale')
    if not 0<=config['anchored_min_minor_fraction']<=.5:raise ValueError('Invalid allele balance bound')
    refinement={r['record_id']:r for r in json.loads(pathlib.Path(args.refinements).read_text())} if args.refinements else {}
    scaffold={};scaffold_samples={}
    if args.scaffold:
        metadata=json.loads(pathlib.Path(args.scaffold).with_suffix('.json').read_text())
        scaffold_chrom=metadata['arguments']['region'].rsplit(':',1)[0]
        with np.load(args.scaffold,allow_pickle=False) as z:
            scaffold_samples={s:i for i,s in enumerate(z['samples'].tolist())}
            gp=z['gp']
            scaffold={(scaffold_chrom,int(p),str(r),str(a)):gp[:,j,:].copy() for j,(p,r,a) in enumerate(zip(z['positions'],z['ref'],z['alt']))}
    source=pysam.VariantFile(args.neural);header=pysam.VariantHeader()
    if args.scaffold and set(source.header.samples)!=set(scaffold_samples):raise ValueError('Neural and scaffold samples disagree')
    header.add_meta('source',value='VarDepthResearch-0.3-experimental')
    header.add_meta('GP_semantics',value='Discriminative arithmetic mixture of read-derived experts; not an exact joint posterior')
    for c in source.header.contigs:header.contigs.add(c,length=source.header.contigs[c].length)
    for name in source.header.samples:header.add_sample(name)
    for name,num,typ,desc in [
        ('GT',1,'String','Unphased diploid genotype'),('GQ',1,'Integer','Phred uncertainty derived from GP; development calibration'),
        ('DP',1,'Integer','Original baseline local molecule count'),('AD','R','Integer','Original allele depths; omitted after haplotype refinement'),
        ('PL','G','Integer','Read-only Phred likelihood; anchored likelihood at refined records, baseline otherwise'),
        ('GP','G','Float','Mixture genotype probabilities in triangular VCF order'),('BGP','G','Float','Original external LD probabilities, zero for newly proposed genotypes'),
        ('NGP','G','Float','Original neural probabilities, zero for newly proposed genotypes'),('DS','A','Float','Alternate dosage from GP'),
        ('CS',1,'Float','Original automatic target-excluded cohort context strength'),('ADP',1,'Integer','Anchored expert molecule count'),
        ('AAD','R','Integer','Exact anchored sequence counts'),('AW',1,'Float','Anchored expert mixture weight'),('SW',1,'Float','SNP scaffold mixture weight')]:
        header.formats.add(name,num,typ,desc)
    header.info.add('SOURCE',1,'String','Original catalog source');header.info.add('REFINED',0,'Flag','Anchored read-derived haplotype expert evaluated')
    header.info.add('NEW_ALLELES',1,'Integer','Read-consensus haplotypes added beyond original neural catalog')
    header.filters.add('LowQual',None,None,'No nonreference sample reaches GQ20')
    records=[];stats=dict(records=0,refined_records=0,added_alleles=0,changed_sample_gt=0,scaffold_samples=0,anchored_samples=0)
    for original in source:
        rid=f'{original.chrom}:{original.pos}:{original.ref}:{",".join(original.alts or [])}';refined=refinement.get(rid)
        alleles=refined['alleles'] if refined else list(original.alleles);n=len(alleles);pairs=order(n);ng=len(pairs)
        position=refined['start0'] if refined else original.start
        item=header.new_record(contig=original.chrom,start=position,stop=position+len(alleles[0]),alleles=alleles)
        item.info['SOURCE']=original.info['SOURCE'] if 'SOURCE' in original.info else 'UNKNOWN'
        if refined:item.info['REFINED']=True;item.info['NEW_ALLELES']=refined['added_alleles'];stats['refined_records']+=1;stats['added_alleles']+=refined['added_alleles']
        indel=any(len(a)!=len(alleles[0]) for a in alleles);passing=False;logref=0
        scaffold_key=(original.chrom,original.start,original.ref,original.alts[0]) if len(original.alleles)==2 and len(original.ref)==len(original.alts[0])==1 else None
        for sample in header.samples:
            c=original.samples[sample];dp=c.get('DP',0);ngp=normalized(project(c['GP'],n));bgp=normalized(project(c.get('BGP',c['GP']),n));gp=ngp.copy()
            if indel:
                weight=config['indel_base_weight']/(1+math.exp(-max(-50,min(50,(dp-config['indel_base_midpoint'])/config['indel_base_width']))))
                gp=(1-weight)*gp+weight*bgp
            sw=0.;aw=0.
            if scaffold_key in scaffold and sample in scaffold_samples:
                # Retain more of the neural cohort signal when another sample
                # supplies a strong target-excluded message.
                cs=float(c.get('CS',0.) or 0.)
                sw=config['scaffold_weight']*math.exp(-dp/config['scaffold_depth_scale'])*(1-min(.5,cs))
                sgp=normalized(scaffold[scaffold_key][scaffold_samples[sample]])
                gp=(1-sw)*gp+sw*sgp;stats['scaffold_samples']+=int(sw>.001)
            out=item.samples[sample]
            # HTSlib needs ploidy before assigning Number=G fields.
            out['GT']=(0,0)
            if refined:
                expert=refined['samples'][sample]
                ea,eb=pairs[int(np.argmax(expert['gp']))]
                balanced=ea==eb or min(expert['exact_ad'][ea],expert['exact_ad'][eb])>=config['anchored_min_minor_fraction']*expert['dp']
                if expert['trusted'] and balanced:
                    aw=config['anchored_weight']*min(1.,max(0.,(dp-15)/15.))
                    gp=(1-aw)*gp+aw*normalized(expert['gp']);stats['anchored_samples']+=int(aw>0)
                out['ADP']=expert['dp'];out['AAD']=tuple(expert['exact_ad']);out['PL']=tuple(min(999,int(round(v))) for v in expert['pl'])
            else:
                if c.get('AD') is not None:out['AD']=c['AD']
                if c.get('PL') is not None:out['PL']=c['PL']
            gp=normalized(gp);best=int(np.argmax(gp));gt=pairs[best];gq=min(99,int(round(-10*math.log10(max(1e-12,1-gp[best])))))
            out['GT']=gt;out['GQ']=gq;out['DP']=dp;out['GP']=tuple(gp);out['BGP']=tuple(bgp);out['NGP']=tuple(ngp)
            out['CS']=float(c.get('CS',0.) or 0.);out['AW']=aw;out['SW']=sw
            out['DS']=tuple(float(sum(p*(int(a==j)+int(b==j)) for p,(a,b) in zip(gp,pairs))) for j in range(1,n))
            stats['changed_sample_gt']+=int(gt!=tuple(c['GT']));passing|=any(gt) and gq>=20;logref+=math.log(max(1e-100,gp[0]))
        item.qual=min(999.,-10/math.log(10)*logref);item.filter.add('PASS' if passing else 'LowQual');records.append(item);stats['records']+=1
    records.sort(key=lambda r:(r.contig,r.start,r.alleles));dest=pathlib.Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    with pysam.VariantFile(str(dest),'wz',header=header) as stream:
        for record in records:stream.write(record)
    pysam.tabix_index(str(dest),preset='vcf',force=True)
    manifest=dict(version='0.3-mixture',config=config,arguments=vars(args),stats=stats,seconds=time.monotonic()-begun,
        code_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),truth_used=False)
    dest.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True);return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--neural',required=True);p.add_argument('--scaffold');p.add_argument('--refinements');p.add_argument('--config');p.add_argument('--output',required=True);run(p.parse_args())
