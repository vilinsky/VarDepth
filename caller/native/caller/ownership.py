"""Two-pass VCF source selection with compact, per-contig haplotype descriptors.

Full VCF records and all alternate genotype arrays are streamed, never retained
for a chromosome. The first pass retains only confident called-haplotype edit
sets within one contig. This preserves long repeat-shift equivalences missed
by grouping on original reference-interval overlap. Memory scales with eligible
called edits on the largest contig, plus the compact removed-index map.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import pysam
from ownership_rules import describe, redundant_records, sha

def components(records):
    group=[];chrom=None;end=-1;last=-1;seen=set()
    for r in records:
        if r.chrom!=chrom:
            if r.chrom in seen:raise ValueError('Contig is not contiguous')
            seen.add(r.chrom)
            if group:yield group
            group=[];chrom=r.chrom;end=-1;last=-1
        if r.start<last:raise ValueError('VCF is not coordinate sorted')
        last=r.start
        if group and r.start>end:
            yield group;group=[];end=-1
        group.append(r.copy());end=max(end,r.stop)
    if group:yield group

def reconcile(input_path,reference_path,output_path,overwrite=False):
    started=time.monotonic();output=Path(output_path)
    if not str(output).endswith(('.vcf','.vcf.gz')):raise ValueError('VCF suffix required')
    if output.resolve()==Path(input_path).resolve():raise ValueError('Original VCF must be preserved')
    if output.exists() and not overwrite:raise ValueError('Preserve earlier output')
    output.parent.mkdir(parents=True,exist_ok=True)
    compressed=str(output).endswith('.gz')
    tmp=output.with_name('.'+uuid.uuid4().hex+('.vcf.gz' if compressed else '.vcf'))
    proof=Path(str(tmp)+'.ownership.jsonl');target_proof=Path(str(output)+'.ownership.jsonl')
    original_sha=sha(input_path);expected=hashlib.sha256()
    totals=dict(input_records=0,retained_records=0,removed_records=0,eligible_records=0,
                components=0,max_component_records=0)
    try:
        removed_all={};descriptions={};hashes={};seen=set();chrom=None;last=-1
        def finish(proofs):
            removed=redundant_records(descriptions);removed_all.update(removed)
            totals['components']+=1;totals['max_component_records']=max(totals['max_component_records'],len(descriptions))
            for i,j in sorted(removed.items()):
                proofs.write(json.dumps(dict(chrom=chrom,removed_index=i,owner_index=j,
                    removed_record_sha256=hashes[i],owner_record_sha256=hashes[j],
                    proof='All sample called diploid haplotypes implied within child scope; both GP >= 0.99'))+'\n')
            descriptions.clear();hashes.clear()
        with (pysam.FastaFile(str(reference_path)) as fa,pysam.VariantFile(str(input_path)) as source,proof.open('w') as proofs):
            samples=list(source.header.samples)
            for i,record in enumerate(source):
                if record.chrom!=chrom:
                    if record.chrom in seen:raise ValueError('Contig is not contiguous')
                    if chrom is not None:finish(proofs)
                    seen.add(record.chrom);chrom=record.chrom;last=-1
                if record.start<last:raise ValueError('VCF is not coordinate sorted')
                last=record.start;totals['input_records']+=1
                fetch=lambda s,e:fa.fetch(chrom,max(0,s),e).upper()
                d=describe(record,samples,fetch)
                if d is not None:
                    descriptions[i]=d;hashes[i]=hashlib.sha256(str(record).encode()).hexdigest()
                    totals['eligible_records']+=1
            if chrom is not None:finish(proofs)
        totals['removed_records']=len(removed_all)
        totals['retained_records']=totals['input_records']-len(removed_all)
        with (pysam.VariantFile(str(input_path)) as source,
              pysam.VariantFile(str(tmp),'wz' if compressed else 'w',header=source.header) as dest):
            for i,record in enumerate(source):
                if i not in removed_all:dest.write(record);expected.update(str(record).encode())
        actual=hashlib.sha256();count=0
        with pysam.VariantFile(str(tmp)) as check:
            for r in check:actual.update(str(r).encode());count+=1
        if actual.digest()!=expected.digest() or count!=totals['retained_records']:raise ValueError('Retained VCF fields changed')
        if sha(input_path)!=original_sha:raise ValueError('Input changed')
        if compressed:pysam.tabix_index(str(tmp),preset='vcf',force=True)
        result=dict(**totals,seconds=time.monotonic()-started,input_sha256=original_sha,
            output_sha256=sha(tmp),proof_sha256=sha(proof),truth_used=False,threshold=.99,
            retained_record_text_identical=True,
            scope='Called-genotype source selection using compact per-contig canonical haplotype descriptors. Original full posterior VCF is preserved; no fusion of alternate catalogs or posterior-calibration claim.')
        os.replace(tmp,output)
        if compressed:os.replace(str(tmp)+'.tbi',str(output)+'.tbi')
        os.replace(proof,target_proof)
        Path(str(output)+'.ownership.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    finally:
        for p in [tmp,Path(str(tmp)+'.tbi'),proof]:p.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(reconcile(a.input,a.reference,a.output)),flush=True)
