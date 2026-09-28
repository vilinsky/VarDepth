"""VarDepth 0.41.0-rc1: frozen Illumina BAM/CRAM to phased chromosome VCF.

Autosomal diploid research release. Call per chromosome using the same release
and concatenate in reference order. The reference panel and map are inputs,
never genotype truth. Clinical certification is outside this release.
"""
import argparse,hashlib,json,os,shutil,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
from measure_stage41 import measure,sha
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['bam','reference','panel','genetic-map','sample','chrom','out']:p.add_argument('--'+n,required=True)
 p.add_argument('--threads',type=int,default=8);p.add_argument('--native-python',default=sys.executable)
 p.add_argument('--phase-workers',type=int,help='Concurrent phase windows; defaults to --threads. Each may use over 2 GiB RAM.')
 p.add_argument('--native-vcf',help='Skip native inference for frozen parity checks only; explicitly recorded')
 p.add_argument('--skip-phase',action='store_true',help='Technical genotype parity check only; not benchmark default')
 a=p.parse_args();assert a.chrom in ['chr'+str(i) for i in range(1,23)] and 1<=a.threads<=32
 if a.phase_workers is None:a.phase_workers=a.threads
 assert 1<=a.phase_workers<=a.threads
 out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);begin=time.monotonic();resources={}
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 temp=out/'tmp';temp.mkdir();env['TMPDIR']=str(temp)
 lock=HERE/'RELEASE_LOCK.json'
 if lock.exists():
  for n,d in json.loads(lock.read_text())['files'].items():assert sha(HERE/n)==d,'Release content changed: '+n
  release_sha=sha(lock)
 else:release_sha=None
 import pysam
 with pysam.FastaFile(a.reference) as f:length=f.get_reference_length(a.chrom)
 with pysam.AlignmentFile(a.bam,reference_filename=a.reference) as f:
  assert a.chrom in f.references and f.get_reference_length(a.chrom)==length
  assert {r.get('SM') for r in f.header.to_dict().get('RG',[])}=={a.sample},'BAM read groups must identify exactly the requested sample'
 with pysam.VariantFile(a.panel) as f:assert a.sample not in f.header.samples,'Target present in panel'
 write(out/'RUN.json',dict(arguments=vars(a),release_lock_sha256=release_sha,truth_input=False,teacher_required=False))
 if a.native_vcf:
  native=Path(a.native_vcf).resolve();write(out/'PARITY_INPUT.json',dict(existing_native=str(native),sha256=sha(native),not_end_to_end_timing=True))
 else:
  folder=out/'native';folder.mkdir();native=folder/'calls.vcf.gz';source=HERE/'native'
  command=[a.native_python,'-B',str(source/'caller/genome.py'),'--checkpoint',str(source/'model/read-model.pt'),'--trust-checkpoint',str(source/'model/fusion-model.pt'),
   '--config',str(source/'selected_config.json'),'--reference',a.reference,'--bam',a.sample+'='+a.bam,'--region',f'{a.chrom}:1-{length}',
   '--panel',a.panel,'--genetic-map',a.genetic_map,'--padding-bp','200000','--chunk-bp','4000000','--cpu-threads',str(a.threads),'--jobs',str(a.threads),
   '--component-jobs','1','--evidence-jobs','1','--device','cpu','--pairhmm-backend','cpu','--compact-neural','--neural-cpu-threads','1',
   '--output',str(native),'--work-dir',str(folder/'work')]
  resources['native_joint_call_impute']=measure(command,out/'resources/native',env)
  pysam.tabix_index(str(native),preset='vcf',force=True)
 sys.path.insert(0,str(HERE/'postlib'))
 from repair36 import domain_mode
 control=[sys.executable,'-B',str(HERE/'control41.py'),'--native',str(native),'--reference',a.reference,'--sample',a.sample,'--chrom',a.chrom,'--out',str(out)]
 resources['coverage_profile']=measure(control+['--stage','profile'],out/'resources/coverage_profile',env)
 guard=json.loads((HERE/'config/GUARD.json').read_text());profile=json.loads((out/'PROFILE.json').read_text())
 correction_needed=any(domain_mode(w,guard)[0]=='current' for w in profile['windows'])
 if correction_needed:
  post=out/'post'
  for stage in ['prepare','reads','predict','confidence']:
   cmd=[sys.executable,'-B',str(HERE/'post41.py'),'--stage',stage,'--native',str(native),'--bam',a.bam,'--reference',a.reference,'--sample',a.sample,'--chrom',a.chrom,'--out',str(post)]
   resources[stage]=measure(cmd,out/'resources'/stage,env)
  current=post/'calibrated-v2/frozen'/a.sample/'calls.vcf.gz'
 else:
  current=native;write(out/'CORRECTION_SKIPPED.json',dict(reason='Every frozen 4Mb guard window selects native; correction cannot affect any emitted record',scientific_decisions_unchanged=True))
 resources['guard_selection']=measure(control+['--stage','select','--current',str(current)],out/'resources/guard_selection',env)
 result=json.loads((out/'guard/COMPLETE.json').read_text())
 genotype=Path(result['policies']['coverage-domain-guard']['output']);final=genotype
 if not a.skip_phase:
  resources['phase_input_identities']=measure(control+['--stage','identities','--bam',a.bam,'--panel',a.panel,'--genetic-map',a.genetic_map],out/'resources/phase_input_identities',env)
  env['VARDEPTH_INPUT_IDENTITIES']=str(out/'PHASE_INPUT_IDENTITIES.json')
  cmd=[sys.executable,'-B',str(HERE/'phase/vardepth_phase39.py'),'--vcf',str(genotype),'--bam',a.bam,'--reference',a.reference,'--panel',a.panel,'--genetic-map',a.genetic_map,
   '--sample',a.sample,'--chrom',a.chrom,'--workers',str(a.phase_workers),'--mode','coverage','--out',str(out/'phase')]
  resources['phasing']=measure(cmd,out/'resources/phasing',env);final=out/'phase/phased.vcf.gz'
 for suffix in ['', '.tbi']:shutil.copyfile(str(final)+suffix,out/('calls.vcf.gz'+suffix))
 write(out/'COMPLETE.json',dict(success=True,version='0.41.0-rc1',release_lock_sha256=release_sha,sample=a.sample,chrom=a.chrom,output=str(out/'calls.vcf.gz'),sha256=sha(out/'calls.vcf.gz'),
  resources=resources,wall_seconds=time.monotonic()-begin,correction_needed=correction_needed,truth_used=False,training=False,phase_enabled=not a.skip_phase,end_to_end=not bool(a.native_vcf)))
 print('VARDP_COMPLETE',a.sample,a.chrom,flush=True)
if __name__=='__main__':main()
