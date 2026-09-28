"""Optional CPU phase stage for a FINAL single-sample VarDepth chromosome VCF.

Uses explicit independent 4 Mb phase windows. Does not join PS across windows,
alter genotypes, or infer parental origin. Requires an unrelated phased panel,
its matching genetic map, an indexed Illumina BAM/CRAM, and the reference FASTA.
"""
import argparse,concurrent.futures,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
import pysam
CODE=Path(__file__).resolve().parent
PHASE_FIELDS={'PS','VPE','VPSRC'}
def invariant(rec):
    calls={}
    for sample,c in rec.samples.items():
        d={k:c[k] for k in c if k not in PHASE_FIELDS}
        if d.get('GT') is not None:d['GT']=sorted(d['GT'],key=lambda x:-1 if x is None else x)
        calls[sample]=d
    return json.dumps([rec.chrom,rec.start,rec.stop,rec.id,rec.alleles,rec.qual,sorted(rec.filter),dict(rec.info),calls],sort_keys=True,separators=(',',':')).encode()+b'\n'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ['vcf','bam','reference','panel','genetic-map','sample','chrom','out']:p.add_argument('--'+field,required=True)
    p.add_argument('--engine',default=str(CODE/'engine'))
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--mode',choices=['conservative','coverage'],default='coverage',help='Coverage mode allows a small observed phase-error tradeoff; conservative retains v38 settings')
    p.add_argument('--start',type=int,default=0,help='Optional 0-based inclusive start')
    p.add_argument('--end',type=int,help='Optional 0-based exclusive end')
    a=p.parse_args()
    settings=json.loads((CODE/'PHASE_MODES.json').read_text())[a.mode]
    if not 1<=a.workers<=32:p.error('workers must be 1..32; memory can exceed 2 GB per worker')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    with pysam.FastaFile(a.reference) as f:length=f.get_reference_length(a.chrom)
    end=a.end if a.end is not None else length
    if not 0<=a.start<end<=length:raise ValueError('Invalid interval')
    bounds=[(x,min(x+4000000,end)) for x in range(a.start,end,4000000)]
    paths=[out/'inputs'/f'w{i:03}.vcf.gz' for i in range(len(bounds))];paths[0].parent.mkdir()
    before=hashlib.sha256();records=0;last=-1
    with pysam.VariantFile(a.vcf) as source:
        if list(source.header.samples)!=[a.sample]:raise ValueError('Exactly one matching sample is required')
        writers=[pysam.VariantFile(str(path),'wz',header=source.header) for path in paths]
        try:
            for rec in source:
                if rec.chrom!=a.chrom or not a.start<=rec.start<end:continue
                if rec.start<last:raise ValueError('VCF must be coordinate sorted')
                last=rec.start;before.update(invariant(rec));records+=1
                writers[(rec.start-a.start)//4000000].write(rec)
        finally:
            for writer in writers:writer.close()
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def one(i):
        lo,hi=bounds[i];target=out/f'w{i:03}'
        cmd=[sys.executable,'-B',str(CODE/'phase39.py'),'--engine',a.engine,'--vcf',str(paths[i]),'--bam',a.bam,
             '--reference',a.reference,'--panel',a.panel,'--genetic-map',a.genetic_map,'--sample',a.sample,
             '--region',f'{a.chrom}:{lo+1}-{hi}','--out',str(target),
             '--max-error',str(settings['max_error']),'--minimum-support',str(settings['minimum_support'])]
        if settings.get('bridge_uninformative'):cmd.append('--bridge-uninformative')
        (out/f'w{i:03}.command.json').write_text(json.dumps(cmd,indent=2))
        with (out/f'w{i:03}.log').open('wb') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,env=environment)
        return target/'baseline.vcf.gz'
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:outputs=list(pool.map(one,range(len(bounds))))
    with pysam.VariantFile(str(outputs[0])) as f:header=f.header.copy()
    header.add_meta('VarDepth_phase_window_policy',value='independent_4Mb_PS;no_cross_window_phase_claim')
    result=out/'phased.vcf.gz'
    with pysam.VariantFile(str(result),'wz',header=header) as writer:
        for path in outputs:
            with pysam.VariantFile(str(path)) as f:
                for rec in f:rec.translate(header);writer.write(rec)
    pysam.tabix_index(str(result),preset='vcf',force=True)
    after=hashlib.sha256();written=phased=0;blocks=set()
    with pysam.VariantFile(str(result)) as f:
        for rec in f:
            after.update(invariant(rec));written+=1;c=rec.samples[a.sample];g=c.get('GT')
            if g and None not in g and len(g)==2 and g[0]!=g[1] and c.phased:
                if c.get('PS') is None:raise AssertionError('Phased heterozygote without PS')
                phased+=1;blocks.add(c['PS'])
    if before.digest()!=after.digest() or records!=written:raise AssertionError('Genotype or evidence invariance failed')
    data=dict(success=True,arguments=vars(a),output=str(result),records=records,phased_heterozygotes=phased,
         phase_blocks=len(blocks),genotype_evidence_sha256=before.hexdigest(),unchanged_unphased_call_set=True,
         output_sha256=hashlib.file_digest(result.open('rb'),'sha256').hexdigest(),wall_seconds=time.monotonic()-begin,
         phase_settings=settings,experimental=True,phase_accuracy_not_guaranteed=True,window_bp=4000000,truth_used=False)
    (out/'COMPLETE.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
if __name__=='__main__':main()
