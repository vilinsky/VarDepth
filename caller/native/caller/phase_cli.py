#!/usr/bin/env python3
"""Experimental BAM + called VCF -> phase-only hybrid QAOA annotation.

Only biallelic heterozygous SNVs are phased. Blocks are bounded and report
discarded cross-block links. Genotypes/GP/DS/GQ are not re-estimated.
"""
from __future__ import annotations
import argparse
import bisect
import collections
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pysam
from quantum_phase import Observation,PhaseProblem,hybrid,exact,export_qasm

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def components(nodes,edges):
    adjacency={i:set() for i in nodes}
    for i,j in edges:
        if i in adjacency and j in adjacency:adjacency[i].add(j);adjacency[j].add(i)
    unseen=set(nodes);result=[]
    while unseen:
        first=min(unseen);unseen.remove(first);queue=[first];block=[]
        while queue:
            i=queue.pop();block.append(i)
            for j in sorted(adjacency[i]&unseen):unseen.remove(j);queue.append(j)
        result.append(sorted(block))
    return result

def extract(vcf,bams,reference,sample,region,*,max_qubits=12,min_gq=20):
    chrom,interval=region.rsplit(':',1);start,end=map(int,interval.split('-'));start-=1
    if start<0 or end<=start or not 2<=max_qubits<=16:raise ValueError('Invalid region or qubit bound')
    sites=[];already_phased=0
    with pysam.VariantFile(vcf) as inp:
        if sample not in inp.header.samples:raise ValueError('Sample absent from VCF')
        for r in inp.fetch(chrom,start,end):
            call=r.samples[sample]
            if len(r.ref)!=1 or len(r.alts or [])!=1 or len(r.alts[0])!=1 or r.ref not in 'ACGT' or r.alts[0] not in 'ACGT':continue
            if call.get('GT') is None or sorted(x for x in call['GT'] if x is not None)!=[0,1]:continue
            if (call.get('GQ') or 0)<min_gq:continue
            if call.phased:already_phased+=1;continue
            sites.append((r.start,r.ref,r.alts[0]))
    # Ambiguous duplicate coordinates are excluded; no arbitrary allele choice.
    counts=collections.Counter(s[0] for s in sites);sites=[s for s in sites if counts[s[0]]==1]
    positions=[s[0] for s in sites];lookup={s[0]:i for i,s in enumerate(sites)}
    molecules={};alignments=0;conflicts=0
    for path in bams:
        with pysam.AlignmentFile(path,reference_filename=reference) as bam:
            groups=bam.header.to_dict().get('RG',[])
            names={r['SM'] for r in groups if r.get('SM')}
            if names!={sample}:raise ValueError('BAM read groups must identify exactly the selected VCF sample')
            if any(r.get('PL','ILLUMINA').upper()!='ILLUMINA' for r in groups):raise ValueError('Phase experiment supports Illumina only')
            rg={r['ID']:r.get('LB',str(Path(path).resolve())+'::'+r['ID']) for r in groups}
            for read in bam.fetch(chrom,start,end):
                if read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate or read.is_qcfail or not 20<=read.mapping_quality<255 or read.query_qualities is None:continue
                if bisect.bisect_left(positions,read.reference_start)==bisect.bisect_left(positions,read.reference_end):continue
                alignments+=1;library=rg.get(read.get_tag('RG'),str(Path(path).resolve())+'::unknown') if read.has_tag('RG') else str(Path(path).resolve())+'::unknown'
                obs=molecules.setdefault((library,read.query_name),{})
                for qpos,rpos in read.get_aligned_pairs(matches_only=True):
                    if rpos not in lookup or qpos is None:continue
                    idx=lookup[rpos];base=read.query_sequence[qpos].upper();q=min(int(read.query_qualities[qpos]),int(read.mapping_quality),60)
                    if q<20 or base not in sites[idx][1:]:continue
                    allele=int(base==sites[idx][2]);error=10**(-q/10.)
                    # Mates are one molecule. Conflicting overlapping bases are
                    # dropped, agreeing bases retain the best single quality.
                    if idx in obs:
                        old=obs[idx]
                        if old is None:continue
                        if old.allele!=allele:obs[idx]=None;conflicts+=1;continue
                        if old.error<=error:continue
                    obs[idx]=Observation(idx,allele,error)
    fragments=[[o for _,o in sorted(obs.items()) if o is not None] for obs in molecules.values()]
    fragments=[f for f in fragments if len(f)>=2]
    complete=PhaseProblem([f'{chrom}:{p+1}:{r}:{a}' for p,r,a in sites],fragments)
    edges=complete.edges();blocks=[]
    for group in components(range(len(sites)),edges):
        for offset in range(0,len(group),max_qubits):
            # Recompute connectivity after splitting; never phase unlinked loci
            # together just because they share a fixed coordinate window.
            for sub in components(group[offset:offset+max_qubits],edges):
                if len(sub)<2:continue
                mapping={j:i for i,j in enumerate(sub)}
                ff=[[Observation(mapping[o.site],o.allele,o.error) for o in f if o.site in mapping] for f in fragments]
                ff=[f for f in ff if len(f)>=2]
                blocks.append((sub,PhaseProblem([complete.sites[j] for j in sub],ff)))
    ownership={j:i for i,(sub,p) in enumerate(blocks) for j in sub}
    return sites,blocks,dict(eligible_heterozygous_snvs=len(sites),molecules_with_links=len(fragments),
        fetched_alignments=alignments,overlap_conflicts=conflicts,blocks=len(blocks),
        cross_block_edges_discarded=sum(ownership.get(i)!=ownership.get(j) for i,j in edges),
        min_gq=min_gq,min_bq=20,min_mq=20,existing_phased_snvs_skipped=already_phased,
        scope='previously unphased biallelic heterozygous SNV; molecule-deduplicated; local phase only')

def run(args):
    started=time.perf_counter();out=Path(args.output)
    if args.phased_vcf and Path(args.phased_vcf).resolve()==Path(args.vcf).resolve():raise ValueError('Output VCF must differ from input')
    out.mkdir(parents=True,exist_ok=True)
    sites,blocks,extraction=extract(args.vcf,args.bam,args.reference,args.sample,args.region,max_qubits=args.max_qubits,min_gq=args.min_gq)
    assignments={};results=[]
    for index,(indices,problem) in enumerate(blocks[:args.max_blocks]):
        if args.backend.startswith('qaoa-sim'):
            result=hybrid(problem,seed=args.seed+index,layers=args.layers,shots=args.shots,max_evaluations=args.max_evaluations,simulator='native' if args.backend=='qaoa-sim-fast' else 'qiskit')
        elif args.backend=='exact-elimination':
            from phase_search import eliminate_unique
            result=dict(eliminate_unique(problem),backend='classical-exact-elimination')
        else:result=dict(exact(problem),backend='classical-exact')
        result.update(sites=problem.sites,fragments=len(problem.fragments),input_index=index)
        if args.backend.startswith('qaoa-sim'):export_qasm(problem,result['quantum']['angles'],out/f'block-{index:04d}.qasm',taper=args.backend=='qaoa-sim-fast')
        # Export sufficient numeric evidence for reproducible solver benchmarks.
        payload=dict(sites=problem.sites,fragments=[[vars(o) for o in f] for f in problem.fragments])
        (out/f'block-{index:04d}.problem.json').write_text(json.dumps(payload)+'\n')
        oracle=result.get('exact',result)
        tied=oracle.get('next_solution_gap') is not None and oracle['next_solution_gap']<=1e-9
        suboptimal=result.get('hybrid_gap_to_exact',0.)>1e-9
        unsolved=result.get('solved',True) is False
        result['phase_applied']=not (tied or suboptimal or unsolved)
        result['abstention_reason']='bounded exact solver did not certify solution' if unsolved else ('ambiguous full-molecule optimum' if tied else ('proposal misses bounded exact optimum' if suboptimal else None))
        results.append(result)
        if result['phase_applied']:
            for j,bit in zip(indices,result['bits']):assignments[sites[j]]=(int(bit),sites[indices[0]][0]+1)
        (out/'progress.json').write_text(json.dumps(dict(completed_blocks=len(results),results=results),indent=2)+'\n')
    if args.phased_vcf:
        if Path(args.phased_vcf).resolve()==Path(args.vcf).resolve():raise ValueError('Output VCF must differ from input')
        chrom=args.region.rsplit(':',1)[0]
        with pysam.VariantFile(args.vcf) as inp:
            header=inp.header.copy()
            if 'PS' not in header.formats:header.formats.add('PS',1,'Integer','Phase set: first 1-based SNV position in local block')
            header.add_meta('VarDepthPhase',value='experimental-r16-'+args.backend+'; phase confidence uncalibrated; GT allele set and GP DS GQ preserved')
            with pysam.VariantFile(args.phased_vcf,'wz' if args.phased_vcf.endswith('.gz') else 'w',header=header) as dest:
                for record in inp:
                    record.translate(header);key=(record.start,record.ref,record.alts[0]) if len(record.alts or [])==1 else None
                    if record.chrom==chrom and key in assignments:
                        bit,ps=assignments[key];call=record.samples[args.sample]
                        call['GT']=(bit,1-bit);call.phased=True;call['PS']=ps
                    dest.write(record)
        if args.phased_vcf.endswith('.gz'):pysam.tabix_index(args.phased_vcf,preset='vcf',force=True)
    report=dict(arguments=vars(args),extraction=extraction,results=results,completed_blocks=len(results),
        skipped_blocks=max(0,len(blocks)-len(results)),phased_sites=len(assignments),seconds=time.perf_counter()-started,
        truth_used=False,hardware_executed=False,quantum_advantage_claimed=False,genotype_quality_reestimated=False,
        input_sha256={p:sha(p) for p in [args.vcf,*args.bam,*([args.reference] if args.reference else [])]})
    (out/'COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['completed_blocks','phased_sites','seconds','hardware_executed']}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vcf',required=True);p.add_argument('--bam',action='append',required=True)
    p.add_argument('--reference');p.add_argument('--sample',required=True);p.add_argument('--region',required=True)
    p.add_argument('--output',required=True);p.add_argument('--phased-vcf')
    p.add_argument('--backend',choices=['qaoa-sim','qaoa-sim-fast','exact','exact-elimination'],default='qaoa-sim')
    p.add_argument('--max-qubits',type=int,default=12);p.add_argument('--max-blocks',type=int,default=16)
    p.add_argument('--min-gq',type=int,default=20);p.add_argument('--shots',type=int,default=1024)
    p.add_argument('--layers',type=int,default=2);p.add_argument('--max-evaluations',type=int,default=40);p.add_argument('--seed',type=int,default=17)
    a=p.parse_args()
    if not 1<=a.max_blocks<=10000:p.error('max-blocks outside limits')
    run(a)

if __name__=='__main__':main()
