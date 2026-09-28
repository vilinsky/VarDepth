"""Replace complex-locus CIGAR observations with whole-haplotype likelihoods."""
import bisect,collections,math
import pysam
from hybrid_phase import allele_from_read,molecular_edges
from haplotype_evidence import observations,Scorer

def edges(bam_path,reference,nodes,sample,chrom,start,end,minimum_support=3):
    bam=pysam.AlignmentFile(str(bam_path),'rc' if str(bam_path).endswith('.cram') else 'rb',reference_filename=str(reference))
    rg={r['ID']:r for r in bam.header.to_dict().get('RG',[])}
    if {v.get('SM') for v in rg.values() if v.get('SM')}!={sample}:raise ValueError('Wrong BAM sample')
    if any(str(v.get('PL','ILLUMINA')).upper()!='ILLUMINA' for v in rg.values()):raise ValueError('Illumina only')
    molecules={};stats=collections.Counter();simple=[(r.start,i) for i,r in enumerate(nodes) if all(len(a)==1 for a in r.alleles)]
    positions=[p for p,i in simple]
    def add(key,i,value):
        group=molecules.setdefault(key,{})
        if i not in group:group[i]=value
        elif group[i] is not None:
            if group[i][0]!=value[0]:group[i]=None;stats['discordant_mate_sites']+=1
            else:group[i]=(value[0],max(value[1],group[i][1]))
    for read in bam.fetch(chrom,start,end):
        if read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate or read.is_qcfail or read.mapping_quality<30 or read.mapping_quality==255:continue
        if read.is_paired and (not read.is_proper_pair or read.next_reference_id!=read.reference_id):continue
        rgid=read.get_tag('RG') if read.has_tag('RG') else '';key=(rg.get(rgid,{}).get('LB',rgid),read.query_name)
        for pos,i in simple[bisect.bisect_right(positions,read.reference_start):bisect.bisect_left(positions,read.reference_end-1)]:
            gt=sorted(nodes[i].samples[sample]['GT']);o=allele_from_read(read,pos,tuple(nodes[i].alleles[a] for a in gt))
            if o is not None:add(key,i,o)
    scorer=Scorer();ref=pysam.FastaFile(str(reference))
    for i,r in enumerate(nodes):
        if all(len(a)==1 for a in r.alleles):continue
        gt=sorted(r.samples[sample]['GT']);indices=sorted({0,*gt})
        rows,meta=observations(r,bam,ref,indices,scorer=scorer);stats['complex_nodes']+=1
        for row in rows:
            allele=indices[row['best']]
            if allele not in gt or row['gap']<math.log(100):continue
            error=max(.02,10**(-min(row['mq'],60)/10),1/(1+math.exp(min(50,row['gap']))))
            add(row['key'],i,(gt.index(allele),error));stats['realigned_observations']+=1
    bam.close();ref.close();stats['informative_molecules']=len(molecules)
    return molecular_edges(molecules.values(),minimum_support),dict(stats)
