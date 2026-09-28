"""Position-free call reliability features; no truth or caller competitor input."""
import collections,math
import numpy as np
NAMES=['GQ','log_DP','log_n_alleles','ref_length','called_max_length','max_length_difference','heterozygous','has_indel','homopolymer','sequence_entropy',
       'GP_called','GP_entropy','NGP_called','BGP_called','PL_called_delta','AD_fraction_called','AD_minor_fraction','AD_missing','ADP_log','AAD_fraction_called','AAD_missing',
       'called_panel_AF_min','called_panel_AF_missing','refined','event_ownership','snv_event_ownership','source_PANEL','source_BOTH']
def _posterior(call,key,index):
    p=call.get(key)
    return float(p[index]) if p and index<len(p) and p[index] is not None and math.isfinite(p[index]) else -1.
def features(r,sample):
    info=dict(r.info);c=r.samples[sample];gt=tuple(sorted(c['GT']));a,b=gt;index=b*(b+1)//2+a;alleles=[r.alleles[j] for j in set(gt)];seq=r.ref
    runs=[];last=None;n=0
    for x in seq:
        if x==last:n+=1
        else:runs.append(n);last=x;n=1
    runs.append(n);counts=collections.Counter(seq);entropy=-sum((n/len(seq))*math.log2(n/len(seq)) for n in counts.values())
    gp=c.get('GP') or ();ge=-sum(float(p)*math.log(max(1e-100,float(p))) for p in gp if p is not None and p>0)
    pl=c.get('PL');pldelta=float(pl[index]-min(x for x in pl if x is not None)) if pl and index<len(pl) and pl[index] is not None else -1.
    def support(key):
        ad=c.get(key)
        if not ad or any(j>=len(ad) or ad[j] is None for j in set(gt)):return -1.,-1.,1.
        total=sum(x for x in ad if x is not None);frac=sum(ad[j] for j in set(gt))/max(1,total);minor=min(ad[j] for j in set(gt))/max(1,total)
        return frac,minor,0.
    ad,minor,missing=support('AD');aad,_,amissing=support('AAD');af=info.get('PANEL_AF') or ();afs=[af[j-1] for j in set(gt) if j>0 and j-1<len(af) and af[j-1] is not None and math.isfinite(af[j-1])]
    x=[float(c.get('GQ') or 0),math.log1p(c.get('DP') or 0),math.log1p(len(r.alleles)),len(seq),max(map(len,alleles)),max(abs(len(x)-len(seq)) for x in alleles),float(a!=b),float(any(len(x)!=len(seq) for x in alleles)),max(runs),entropy,
       _posterior(c,'GP',index),ge,_posterior(c,'NGP',index),_posterior(c,'BGP',index),pldelta,ad,minor,missing,math.log1p(c.get('ADP') or 0),aad,amissing,
       min(afs) if afs else -1.,float(not afs),float('REFINED' in info),float('EVENT_OWNERSHIP' in info),float('SNV_EVENT_OWNERSHIP' in info),float(info.get('SOURCE')=='PANEL'),float(info.get('SOURCE')=='BOTH')]
    if len(x)!=len(NAMES) or not np.isfinite(x).all():raise ValueError('Invalid reliability features')
    return np.asarray(x,np.float32)


def score_records(model,records,sample):
    """Batch tree inference without changing the feature vector or threshold."""
    if not records:return np.empty(0,float)
    return model.predict_proba(np.stack([features(r,sample) for r in records]))[:,1]


def apply_vcf(model_path,manifest_path,vcf,sample,output,region,batch_size=4096):
    """Apply an explicitly selected, hash-verified development model.

    Pickle is loaded only after matching the supplied trusted model manifest.
    Neither file should be accepted from an untrusted party. GT abstention is
    separate from GP/DS/GQ, which remain the original caller evidence.
    """
    import hashlib,json,pickle,time
    from pathlib import Path
    import pysam
    begin=time.monotonic();cpu=time.process_time();out=Path(output)
    if out.exists() or Path(str(out)+'.json').exists():raise FileExistsError(out)
    if batch_size<1:raise ValueError('Empty inference batch')
    manifest=json.loads(Path(manifest_path).read_text())
    blob=Path(model_path).read_bytes();digest=hashlib.sha256(blob).hexdigest()
    if digest!=manifest['model_sha256'] or manifest['features']!=NAMES:raise ValueError('Model identity differs')
    model=pickle.loads(blob)
    if list(model.classes_)!=[0,1]:raise ValueError('Unsupported target labels')
    chrom,span=region.split(':');start,end=map(int,span.split('-'));start-=1
    if start<0 or end<=start:raise ValueError('Invalid 1-based inclusive interval')
    source=pysam.VariantFile(vcf);header=source.header.copy()
    if sample not in header.samples:raise ValueError('Sample missing')
    if 'R16PC' not in header.formats:header.formats.add('R16PC',1,'Float','Development call reliability; not genotype posterior')
    threshold=manifest['threshold'];stats=collections.Counter();buffer=[]
    out.parent.mkdir(exist_ok=True,parents=True)
    with pysam.VariantFile(str(out),'wz' if str(out).endswith('.gz') else 'w',header=header) as dest:
        def flush():
            scored=[]
            for r in buffer:
                gt=r.samples[sample].get('GT')
                if gt and len(gt)==2 and None not in gt and any(gt):scored.append(r)
            probabilities=iter(score_records(model,scored,sample));wanted={id(r) for r in scored}
            for r in buffer:
                r.translate(header);c=r.samples[sample]
                if id(r) in wanted:
                    p=float(next(probabilities));c['R16PC']=p;stats['scored']+=1
                    if p<threshold:c['GT']=(None,None);c.phased=False;stats['abstained']+=1
                dest.write(r);stats['records']+=1
            buffer.clear()
        for r in source.fetch(chrom,start,end):
            # Do not include a left-overlapping source record outside scope.
            if not start<=r.start<end:continue
            buffer.append(r.copy())
            if len(buffer)>=batch_size:flush()
        if buffer:flush()
    source.close()
    if str(out).endswith('.gz'):pysam.tabix_index(str(out),preset='vcf',force=True)
    result=dict(exploratory=True,production_promoted=False,model_sha256=digest,threshold=threshold,
                sample=sample,region=region,stats=dict(stats),wall_seconds=time.monotonic()-begin,
                cpu_seconds=time.process_time()-cpu,GP_DS_GQ_preserved=True)
    Path(str(out)+'.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser(description='Experimental 30x call-reliability abstention; model is not clinically validated')
    for key in ['model','manifest','vcf','sample','output','region']:p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(apply_vcf(a.model,a.manifest,a.vcf,a.sample,a.output,a.region)))
