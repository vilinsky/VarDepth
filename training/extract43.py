"""Join current v41 query features to hap.py labels, entirely on research VM."""
import argparse,collections,concurrent.futures,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import pysam
from features43 import key_and_kind,features,depth_profile,NAMES

ROOT=Path('/mnt/vdpub14-analysis01/v41-release-benchmark')
OUT=Path('/mnt/vdpub14-analysis01/v43-filter')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def split(chrom):
    n=int(chrom[3:]);return 'train' if n<=12 else 'calibration' if n<=16 else 'test'
def one(task):
    case,chrom=task;begin=time.monotonic();folder=ROOT/'assessment/vardepth'/case/chrom
    out=OUT/'features'/case/chrom;out.mkdir(parents=True,exist_ok=False)
    blocked=set(json.loads((Path(__file__).parent/'blocked.json').read_text()))
    adapter=json.loads((folder/'ADAPTER.json').read_text());source=folder/'query.vcf.gz';happy=folder/'ALL/happy.vcf.gz'
    assert sha(source)==adapter['query_sha256'] and adapter['query_sha256'] not in blocked
    happy_sha=sha(happy);assert happy_sha not in blocked
    labels={};stats=collections.Counter();hap_types=collections.Counter()
    with pysam.VariantFile(happy) as f:
        assert list(f.header.samples)==['TRUTH','QUERY']
        for r in f:
            q=r.samples['QUERY'];decision=q.get('BD');bvt=q.get('BVT')
            if decision not in ['TP','FP'] or bvt not in ['SNP','INDEL']:continue
            stats['eligible_happy_decisions']+=1;hap_types[bvt+'_'+decision]+=1
            key,kind=key_and_kind(r,'QUERY')
            if key is None or kind!=int(bvt=='INDEL'):stats['unsupported_happy_decision']+=1;continue
            region=set(r.info.get('Regions',()));bits=sum(n for label,n in [('capture',1),('flank100',2),('outside',4)] if label in region)
            if key in labels:labels[key]=None;stats['ambiguous_happy_key']+=1
            else:labels[key]=(int(decision=='TP'),kind,bits)
    profile=depth_profile(source);xs=[];ys=[];kinds=[];regions=[];oldpass=[];depths=[]
    with pysam.VariantFile(source) as f:
        sample=list(f.header.samples)[0]
        for r in f:
            stats['query_records']+=1;key,kind=key_and_kind(r,sample)
            if key is None:stats['unsupported_query']+=1;continue
            target=labels.get(key)
            if target is None:stats['unmatched_or_ambiguous_query']+=1;continue
            if target[1]!=kind:stats['type_mismatch']+=1;continue
            stats['matched_'+('SNV' if kind==0 else 'INDEL')+'_'+('TP' if target[0] else 'FP')]+=1
            # Fixed, outcome-independent sampling applies only to training.
            # Same locus/GT across assays has the same inclusion decision.
            if split(chrom)=='train':
                cutoff=.2 if kind==0 else .5
                value=int.from_bytes(hashlib.blake2b(repr(key).encode(),digest_size=8).digest(),'big')/2**64
                if value>=cutoff:continue
            filt=set(r.filter)
            if filt-set(['PASS','.','LowQual']):stats['hard_filtered_excluded']+=1;continue
            x=features(r,sample,profile.get((r.chrom,r.start//50000),float('nan')))
            assert len(x)==len(NAMES)
            xs.append(x);ys.append(target[0]);kinds.append(kind);regions.append(target[2]);oldpass.append(not (filt-{'PASS','.'}))
            depths.append(r.samples[sample].get('DP') if r.samples[sample].get('DP') is not None else -1)
    np.savez_compressed(out/'data.npz',x=np.asarray(xs,np.float32).reshape(-1,len(NAMES)),y=np.asarray(ys,np.int8),
        kind=np.asarray(kinds,np.int8),region=np.asarray(regions,np.int8),oldpass=np.asarray(oldpass,bool),dp=np.asarray(depths,np.int32))
    report=dict(success=True,case=case,chrom=chrom,split=split(chrom),rows=len(xs),stats=dict(stats),happy_decisions=dict(hap_types),
        feature_names=NAMES,source_query_sha256=adapter['query_sha256'],happy_sha256=happy_sha,wall_seconds=time.monotonic()-begin,
        training_sampling={'SNV':.2,'INDEL':.5} if split(chrom)=='train' else None,
        genomic_data_exported=False,labels_are_genotype_query_correctness=True)
    (out/'COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(case=case,chrom=chrom,rows=len(xs),matched=sum(n for k,n in stats.items() if k.startswith('matched_')),success=True)),flush=True)
    return report
def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','test'],required=True);a=p.parse_args()
    tasks=[]
    for case in ['lowpass','wes','hybrid','wgs']:
        for n in range(1,23):
            chrom='chr'+str(n)
            if (n>=17)!=(a.split=='test'):continue
            folder=ROOT/'assessment/vardepth'/case/chrom
            if (folder/'COMPLETE.json').exists():tasks.append((case,chrom))
    if a.split=='test':assert (OUT/'MODEL_FROZEN.json').exists(),'Freeze model before opening test labels'
    OUT.mkdir(exist_ok=True)
    start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(one,tasks))
    (OUT/('EXTRACTION_'+a.split.upper()+'.json')).write_text(json.dumps(dict(success=True,results=results,wall_seconds=time.monotonic()-start),indent=2)+'\n')
if __name__=='__main__':main()
