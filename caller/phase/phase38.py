"""Genotype-preserving experimental phase audit and scaffold extension."""
import argparse,collections,hashlib,json,sys,time
from pathlib import Path

def save(path,data):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(data,indent=2)+'\n')

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['engine','vcf','bam','reference','sample','region','panel','genetic-map','out']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--min-af',type=float,default=.001)
    a=p.parse_args();sys.path.insert(0,a.engine)
    import pysam
    from hybrid_phase import Edge,phase_forest,molecular_edges
    from hybrid_phase_cli import panel_scaffold
    from collect_molecules38 import collect
    from scaffold_attach38 import attach,overlapping_indices
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    begin=time.monotonic();cpu=time.process_time()
    chrom,interval=a.region.split(':');start,end=map(int,interval.split('-'));start-=1
    assert 0<end-start<=5_000_000
    with pysam.VariantFile(a.vcf) as f:
        assert list(f.header.samples)==[a.sample]
        header=f.header.copy();records=[r.copy() for r in f]
    eligible=[];hets=[]
    for r in records:
        if r.contig!=chrom or not start<=r.start<end:continue
        c=r.samples[a.sample];g=c.get('GT')
        if g is None or len(g)!=2 or None in g or (c.get('GQ') or 0)<20:continue
        if any(x<0 or x>=len(r.alleles) for x in g) or any(set(x.upper())-set('ACGT') for x in r.alleles):continue
        eligible.append(r)
        if g[0]!=g[1] and (not r.filter.keys() or list(r.filter)==['PASS']):hets.append(r)
    bad=overlapping_indices(hets)
    # Keep an explicit audit of the old adjacency-only overlap detector.
    old_bad=set()
    for i in range(len(hets)-1):
        if hets[i].stop>hets[i+1].start:old_bad.update([i,i+1])
    nodes=[r for i,r in enumerate(hets) if i not in bad];ids={id(r):i for i,r in enumerate(nodes)}
    proposed,meta=panel_scaffold(a.panel,eligible,a.sample,set(header.samples),chrom,start,end,a.min_af,64,a.genetic_map,a.reference,True)
    edges=[]
    for e in proposed:
        left=ids.get(id(eligible[e.left]));right=ids.get(id(eligible[e.right]))
        if left is not None and right is not None:edges.append(Edge(left,right,e.parity,e.error,e.source))
    molecules,read_meta=collect(a.bam,a.reference,nodes,a.sample,chrom,start,end,3)
    read_links=molecular_edges(molecules.values(),3);edges+=read_links
    baseline,forest=phase_forest(len(nodes),edges,.01)
    # Do not undo the original conflict quarantine or its path-budget split.
    # Only sites lacking any previously qualifying edge may be attached.
    forbidden={i for e in edges if e.error<=.01 for i in (e.left,e.right)}-set(baseline)
    t=time.process_time();improved,extension,details=attach(baseline,molecules.values(),.01,3,forbidden)
    extension['quarantine_or_budget_exclusions']=len(forbidden)
    extension_cpu=time.process_time()-t
    for name,number,typ,desc in [('PS',1,'Integer','Phase set identifier'),
         ('VPE',1,'Float','Model maximum pair-path error bound; not empirically calibrated'),
         ('VPSRC',1,'String','Experimental phase source')]:
        if name not in header.formats:header.formats.add(name,number,typ,desc)
    header.add_meta('VarDepth_phase_v38',value='research;fixed_genotypes;one_molecule_one_target_block_vote')
    manifests={};attached={d['node'] for d in details}
    for arm,assignments in [('baseline',baseline),('extended',improved)]:
        dst=out/(arm+'.vcf.gz');by_id={id(nodes[i]):(i,value) for i,value in assignments.items()};stats=collections.Counter()
        with pysam.VariantFile(str(dst),'wz',header=header) as writer:
            for r in records:
                key=id(r);r=r.copy();r.translate(header);c=r.samples[a.sample]
                if r.contig==chrom and start<=r.start<end:
                    c.phased=False
                    for field in ['PS','VPE','VPSRC']:
                        if field in c:c[field]=None
                    # These allele-oriented fields need explicit handling before
                    # supporting an input that contains them.
                    if any(c.get(k) is not None for k in ['PGT','PID','HP','HD','HQ'] if k in c):
                        raise ValueError('Unsupported inherited haplotype annotation')
                    if key in by_id:
                        i,(root,bit,error)=by_id[key];gt=sorted(c['GT']);c['GT']=tuple(gt if bit==0 else gt[::-1])
                        c.phased=True;c['PS']=nodes[root].pos;c['VPE']=error
                        c['VPSRC']='SCAFFOLD_ATTACH' if arm=='extended' and i in attached else 'HYBRID'
                        stats['phased_hets']+=1
                    else:stats['unphased_records']+=1
                writer.write(r)
        pysam.tabix_index(str(dst),preset='vcf',force=True)
        manifests[arm]=dict(path=str(dst),sha256=sha(dst),counts=dict(stats))
    node_details=[dict(index=i,pos1=r.pos,alleles=r.alleles,gt=r.samples[a.sample]['GT'],GQ=r.samples[a.sample].get('GQ')) for i,r in enumerate(nodes)]
    save(out/'GRAPH.json',dict(nodes=node_details,edges=[vars(e) for e in edges],baseline={str(k):v for k,v in baseline.items()},attachments=details))
    save(out/'MOLECULES.json',[[[i,list(v) if v is not None else None] for i,v in sorted(m.items())] for m in molecules.values()])
    save(out/'COMPLETE.json',dict(success=True,arguments=vars(a),source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
         inputs={k:dict(path=getattr(a,k),sha256=sha(getattr(a,k))) for k in ['vcf','bam','reference','panel','genetic_map']},
         panel=meta,reads=read_meta,forest=forest,extension=extension,extension_cpu_seconds=extension_cpu,
         eligible_hets=len(hets),overlap_hets=len(bad),additional_nested_overlap_exclusions=len(bad-old_bad),outputs=manifests,wall_seconds=time.monotonic()-begin,cpu_seconds=time.process_time()-cpu,
         truth_used=False,old_phase_used=False,genotype_training=False))
    print(json.dumps(dict(sample=a.sample,extension=extension,outputs=manifests)),flush=True)

if __name__=='__main__':main()
