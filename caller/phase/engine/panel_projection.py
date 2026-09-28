"""Truth-free projection of called haplotypes onto typed panel sequence alleles."""
import collections,tempfile
from pathlib import Path
import pysam
import pysam.bcftools as bc

def project(calls,sample,reference,chrom):
    """Return unique normalized proxies with orientation relative to sorted GT.

    An atom is a phase observation of its original record, not a new genotype.
    Conflicting duplicate keys are withheld rather than choosing an owner.
    """
    with tempfile.TemporaryDirectory(prefix='vardepth-panel-projection-') as tmp:
        raw=Path(tmp)/'input.vcf';norm=Path(tmp)/'atoms.vcf'
        header=pysam.VariantHeader();header.contigs.add(chrom)
        header.formats.add('GT',1,'String','Ordered original called haplotypes');header.add_sample(sample)
        with pysam.VariantFile(str(raw),'w',header=header) as out:
            for i,r in enumerate(calls):
                gt=sorted(r.samples[sample]['GT']);chosen=sorted({0,*gt})
                if len(chosen)==1:chosen=list(range(len(r.alleles)))
                if len(chosen)<2:continue
                v=out.new_record(contig=chrom,start=r.start,id=str(i),alleles=[r.alleles[a] for a in chosen])
                v.samples[sample]['GT']=tuple(chosen.index(a) for a in gt);out.write(v)
        bc.norm('-a','--atom-overlaps','.','-m','-any','-f',str(reference),'-Ov','-o',str(norm),str(raw),catch_stdout=False)
        entries=collections.defaultdict(set);stats=collections.Counter()
        for r in pysam.VariantFile(str(norm)):
            gt=r.samples[sample]['GT']
            if len(r.alleles)!=2 or gt is None or None in gt or set(gt)-{0,1}:continue
            if set(''.join(r.alleles))-set('ACGT') or abs(len(r.ref)-len(r.alts[0]))>50:continue
            k=(r.start,r.ref,r.alts[0]);entries[k].add((int(r.id),sum(gt),gt[0]))
    proxies={}
    for k,v in entries.items():
        if len(v)!=1:stats['ambiguous_proxies']+=1;continue
        proxies[k]=next(iter(v));stats['SNV' if len(k[1])==len(k[2])==1 else 'indel']+=1
    return proxies,dict(stats)

def original_edge(a,b,parity):
    """Map proxy REF orientations to original sorted-haplotype orientations."""
    ia,ga,oa=a;ib,gb,ob=b
    if ga!=1 or gb!=1:raise ValueError('Phase endpoints must be heterozygotes')
    return ia,ib,int(parity)^int(oa)^int(ob)
