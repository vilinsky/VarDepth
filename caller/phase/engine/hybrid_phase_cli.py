"""Finite-region integrated phasing pilot; preserves every input genotype."""
import argparse,bisect,collections,hashlib,json,math,tempfile,time
from pathlib import Path
import numpy as np
import pysam
from hybrid_phase import Edge,phase_edges,phase_forest,allele_from_read,molecular_edges

FAMILIES=[{'HG001'}|{'NA'+str(n) for n in range(12877,12894)},
 {'HG002','HG003','HG004','NA24385','NA24149','NA24143'},
 {'HG005','HG006','HG007','NA24631','NA24694','NA24695'}]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def choose_donors(h,g,positions,count):
    """Deterministic genotype-compatible paired proposals across eight windows."""
    count=min(count,h.shape[1]);selected=[];seen=set();proposals=[]
    def add(i):
        i=int(i)
        if i not in seen and len(selected)<count:seen.add(i);selected.append(i)
    limits=np.linspace(positions[0],positions[-1]+1,9)
    windows=[np.ones(len(g),bool)]+[(positions>=a)&(positions<b) for a,b in zip(limits[:-1],limits[1:])]
    for mask in windows:
        if not mask.any():continue
        a=h[mask].astype(float);d=g[mask]
        # A donor allele is compatible with either chromosome at a het.
        score=(np.where(d==0,1.,np.where(d==2,-1.,0.)))@a
        ranked=np.argsort(score,kind='stable');local=[]
        for seed in ranked[:2]:
            target=d-a[:,seed]
            pair_score=a.sum(0)-2*(target@a)
            local.append((int(seed),np.argsort(pair_score,kind='stable')[:count]))
        proposals.append(local)
    for rank in range(count):
        for local in proposals:
            for seed,partners in local:add(seed);add(partners[rank])
        if len(selected)==count:break
    return np.asarray(selected)

def panel_scaffold(path,calls,sample,excluded,chrom,start,end,min_af,states,map_path,reference=None,projected=False):
    excluded=set(filter(None,excluded))|{sample}
    for family in FAMILIES:
        if family&excluded:excluded|=family
    f=pysam.VariantFile(str(path));names=[s for s in f.header.samples if s not in excluded]
    if len(names)<2:raise ValueError('Insufficient unrelated reference donors')
    projection_meta={}
    if projected:
        from panel_projection import project
        lookup,projection_meta=project(calls,sample,reference,chrom)
    else:
        candidates=[(i,r) for i,r in enumerate(calls) if len(r.alleles)==2 and len(r.ref)==len(r.alts[0])==1]
        multiplicity=collections.Counter(r.start for i,r in candidates)
        lookup={(r.start,r.ref,r.alts[0]):(i,sum(r.samples[sample]['GT']),sorted(r.samples[sample]['GT'])[0]) for i,r in candidates if multiplicity[r.start]==1}
    positions=[];rows=[];genotypes=[];indices=[];skips=collections.Counter()
    for r in f.fetch(chrom,start,end):
        if len(r.alleles)!=2:continue
        key=(r.start,r.ref,r.alts[0])
        if key not in lookup:continue
        i,dosage,orientation=lookup[key]
        af=r.info.get('AF',(None,))[0]
        if af is not None and min(af,1-af)<min_af:continue
        raw=[];valid=True
        for name in names:
            s=r.samples[name];gt=s.get('GT')
            if gt is None or len(gt)!=2 or None in gt or not s.phased or not all(a in (0,1) for a in gt):valid=False;break
            raw.extend(gt)
        if not valid:skips['incomplete_or_unphased_panel_site']+=1;continue
        row=np.asarray(raw,np.uint8)
        if min(row.mean(),1-row.mean())<min_af:continue
        positions.append(r.start);rows.append(row);genotypes.append(dosage);indices.append((i,dosage,orientation))
    f.close()
    if len(rows)<2:return [],dict(sites=len(rows),excluded=sorted(excluded),skips=dict(skips))
    h=np.stack(rows);g=np.asarray(genotypes);pos=np.asarray(positions)
    keep=choose_donors(h,g,pos,states)
    # Read only supplied genetic-map data; no truth or outcome-selected gaps.
    import sys
    sys.path.insert(0,str(Path(__file__).parent/'base'))
    from genetic_map import transitions
    refresh=transitions(pos,reference_haplotypes=h.shape[1],path=map_path,chrom=chrom)
    result=[]
    for a,b,p in phase_edges(h[:,keep],g,refresh):
        from panel_projection import original_edge
        ia,ib,parity=original_edge(indices[a],indices[b],int(p>.5))
        if ia!=ib:result.append(Edge(ia,ib,parity,min(p,1-p),'panel'))
    return result,dict(sites=len(rows),panel_samples=len(names),donor_haplotypes=len(keep),
        selected_haplotypes=keep.tolist(),excluded=sorted(excluded),skips=dict(skips),projection=projection_meta)

def read_edges(bam_path,reference,nodes,sample,chrom,start,end,minimum_support=3):
    bam=pysam.AlignmentFile(str(bam_path),'rc' if str(bam_path).endswith('.cram') else 'rb',reference_filename=str(reference))
    rg={r['ID']:r for r in bam.header.to_dict().get('RG',[])}
    sms={v.get('SM') for v in rg.values() if v.get('SM')}
    if sms!={sample}:raise ValueError('BAM sample must match the single target explicitly')
    if any(str(v.get('PL','ILLUMINA')).upper()!='ILLUMINA' for v in rg.values()):raise ValueError('Illumina only')
    # Project only the two called haplotypes to primitive observation sites.
    # A molecule can cover one informative part of a compound allele. Daughter
    # GT ordering maps observations to the original diploid node's orientation.
    import pysam.bcftools as bc
    units=[]
    with tempfile.TemporaryDirectory(prefix='vardepth-phase-') as directory:
        raw=Path(directory)/'source.vcf';atoms=Path(directory)/'observations.vcf'
        header=pysam.VariantHeader();header.contigs.add(chrom)
        header.formats.add('GT',1,'String','Ordered original heterozygous genotype');header.add_sample(sample)
        with pysam.VariantFile(str(raw),'w',header=header) as writer:
            for i,node in enumerate(nodes):
                gt=sorted(node.samples[sample]['GT']);chosen=sorted({0,*gt})
                record=writer.new_record(contig=chrom,start=node.start,id=str(i),alleles=[node.alleles[j] for j in chosen])
                record.samples[sample]['GT']=tuple(chosen.index(a) for a in gt);writer.write(record)
        bc.norm('-a','--atom-overlaps','.','-m','-any','-f',str(reference),'-Ov','-o',str(atoms),str(raw),catch_stdout=False)
        with pysam.VariantFile(str(atoms)) as f:
            for r in f:
                gt=r.samples[sample]['GT']
                if (len(r.alleles)==2 and gt and set(gt)=={0,1} and
                    not(set(''.join(r.alleles).upper())-set('ACGT')) and abs(len(r.ref)-len(r.alts[0]))<=50):
                    units.append((r.start,int(r.id),r.alleles,gt))
    units=sorted(set(units));pos=[u[0] for u in units];molecules={};counts=collections.Counter(observation_sites=len(units))
    for read in bam.fetch(chrom,start,end):
        counts['seen_reads']+=1
        if (read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate or
            read.is_qcfail or read.mapping_quality<30 or read.mapping_quality==255):continue
        if read.is_paired and (not read.is_proper_pair or read.next_reference_id!=read.reference_id):continue
        left=bisect.bisect_right(pos,read.reference_start);right=bisect.bisect_left(pos,read.reference_end-1)
        rgid=read.get_tag('RG') if read.has_tag('RG') else ''
        library=rg.get(rgid,{}).get('LB',rgid)
        # True LB plus QNAME combines mates; missing LB falls back to RG, never
        # combines libraries based only on equal read names.
        key=(library,read.query_name)
        for unit in units[left:right]:
            position,i,alleles,gt=unit;obs=allele_from_read(read,position,alleles)
            if obs is None:continue
            if obs[0] not in gt:continue
            allele=gt.index(obs[0]);value=(allele,obs[1]);group=molecules.setdefault(key,{})
            if i not in group:group[i]=value
            elif group[i] is not None:
                if group[i][0]!=allele:group[i]=None;counts['discordant_mate_sites']+=1
                else:group[i]=(allele,max(group[i][1],value[1]))
    bam.close();counts['informative_molecules']=len(molecules)
    return molecular_edges(molecules.values(),minimum_support),dict(counts)

def run(args):
    begin=time.monotonic();cpu=time.process_time();out=Path(args.output)
    if out.exists() or Path(str(out)+'.json').exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    chrom,interval=args.region.split(':');start,end=map(int,interval.split('-'));start-=1
    if end<=start or end-start>5_000_000:raise ValueError('Pilot requires an explicit interval of at most 5 Mb')
    f=pysam.VariantFile(args.vcf)
    if args.sample not in f.header.samples:raise ValueError('Target absent from VCF')
    header=f.header.copy();records=[r.copy() for r in f];f.close()
    eligible=[];hets=[];skips=collections.Counter()
    for r in records:
        s=r.samples[args.sample];g=s.get('GT')
        if r.contig!=chrom or not start<=r.start<end:continue
        if (g is None or len(g)!=2 or None in g or (s.get('GQ') or 0)<args.min_gq or
            any(a<0 or a>=len(r.alleles) for a in g) or
            any(not a or set(a.upper())-set('ACGT') for a in r.alleles)):continue
        eligible.append(r)
        if g[0]!=g[1] and (not r.filter.keys() or list(r.filter.keys())==['PASS']):hets.append(r)
    # Ambiguous/overlapping representations need haplotype realignment, not a
    # forced assumption that separate records are independent loci.
    bad=set()
    for i in range(len(hets)-1):
        if hets[i].stop>hets[i+1].start:bad|={id(hets[i]),id(hets[i+1])}
    nodes=[r for r in hets if id(r) not in bad];node_ids={id(r):i for i,r in enumerate(nodes)}
    edges=[];panel_meta={'disabled':True}
    if args.panel:
        excluded=set(args.exclude_samples.split(','))|set(header.samples)
        proposed,panel_meta=panel_scaffold(args.panel,eligible,args.sample,excluded,chrom,start,end,args.min_af,args.states,args.genetic_map,args.reference,getattr(args,'panel_projection',False))
        for e in proposed:
            a=node_ids.get(id(eligible[e.left]));b=node_ids.get(id(eligible[e.right]))
            if a is not None and b is not None:edges.append(Edge(a,b,e.parity,e.error,e.source))
    read_function=read_edges
    if getattr(args,'realign_phase',False):
        from realign_phase import edges as read_function
    read_links,read_meta=read_function(args.bam,args.reference,nodes,args.sample,chrom,start,end,getattr(args,'minimum_read_support',3));edges+=read_links
    assignments,forest=phase_forest(len(nodes),edges,args.block_error)
    for name,number,typ,desc in [
        ('PS',1,'Integer','Phase set identifier'),
        ('VPE',1,'Float','Experimental maximum pair-path union error bound in block; not empirically calibrated'),
        ('VPSRC',1,'String','Experimental panel/read graph phase annotation')]:
        if name not in header.formats:header.formats.add(name,number,typ,desc)
    header.add_line('##vardepth_hybrid_phase=experimental_v16_GT_GP_DS_GQ_unchanged')
    by_id={id(nodes[i]):value for i,value in assignments.items()};phased=collections.Counter();spans={}
    with pysam.VariantFile(str(out),'wz' if str(out).endswith('.gz') else 'w',header=header) as writer:
        for r in records:
            key=id(r);r.translate(header);s=r.samples[args.sample]
            if r.contig==chrom and start<=r.start<end:
                # Recompute phase as a whole; inherited phase is not evidence.
                s.phased=False
                if 'PS' in s:s['PS']=None
                if key in by_id:
                    root,bit,error=by_id[key];a,b=sorted(s['GT']);s['GT']=(a,b) if bit==0 else (b,a)
                    s.phased=True;s['PS']=nodes[root].pos;s['VPE']=error;s['VPSRC']='HYBRID'
                    kind='SNV' if all(len(a)==1 for a in r.alleles) else 'indel';phased[kind]+=1
                    spans.setdefault(nodes[root].pos,[]).append(r.pos)
            writer.write(r)
    if str(out).endswith('.gz'):pysam.tabix_index(str(out),preset='vcf',force=True)
    report=dict(experimental=True,empirically_calibrated=False,arguments=vars(args),
        input_sha256={p:sha(p) for p in [args.vcf,args.bam,args.reference]+([args.panel,args.genetic_map] if args.panel else [])},
        source_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('hybrid_phase.py'),Path(__file__).with_name('panel_projection.py'),Path(__file__).with_name('realign_phase.py'),Path(__file__).with_name('haplotype_evidence.py')]},
        eligible_hets=len(hets),overlap_hets_withheld=len(bad),phased=dict(phased),panel=panel_meta,reads=read_meta,
        edges=dict(collections.Counter(e.source for e in edges)),forest=forest,
        maximum_block_span=max((max(p)-min(p)+1 for p in spans.values()),default=0),
        cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-begin)
    Path(str(out)+'.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['arguments','input_sha256','source_sha256','panel']}),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['vcf','bam','reference','sample','region','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--panel-projection',action='store_true');p.add_argument('--realign-phase',action='store_true');p.add_argument('--panel');p.add_argument('--genetic-map');p.add_argument('--exclude-samples',default='')
    p.add_argument('--minimum-read-support',type=int,choices=range(2,11),default=3)
    p.add_argument('--min-gq',type=int,default=20);p.add_argument('--min-af',type=float,default=.01)
    p.add_argument('--states',type=int,default=64);p.add_argument('--block-error',type=float,default=.01)
    args=p.parse_args()
    if args.panel and not args.genetic_map:p.error('Panel mode requires a genetic map')
    if args.states<2 or args.states>128:p.error('Pilot supports 2 to 128 donor states')
    run(args)

if __name__=='__main__':main()
