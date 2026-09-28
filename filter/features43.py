"""Truth-free, position-free features for coverage-adaptive acceptance.

Chromosome/position are used only for joining labels and local depth lookup;
neither, nor sample/assay/capture membership, is passed to the classifier.
"""
import collections
import math
import numpy as np

NAMES = ['log_DP','log_ADP','log_context_DP','depth_context_ratio','GQ','log_QUAL',
         'GP_max','GP_margin','GP_entropy','GP_ref','NGP_max','BGP_max',
         'log_AD_total','AD_balance_error','log_AAD_total','AAD_balance_error',
         'PL_margin','log_allele_length','log_indel_size','heterozygous',
         'panel_AF_min','panel_AF_max','source_panel','source_reads','refined',
         'log_added_alleles','correction_PC','correction_PV','source_corrected']

def number(x):
    try:
        x=float(x)
        return x if math.isfinite(x) else float('nan')
    except (TypeError,ValueError):return float('nan')

def lognum(x):
    x=number(x)
    return math.log1p(max(0.,x)) if math.isfinite(x) else x

def vector(x):
    if x is None:return []
    if not isinstance(x,(list,tuple,np.ndarray)):x=[x]
    return [number(a) for a in x]

def key_and_kind(rec, sample):
    gt=rec.samples[sample].get('GT')
    if not gt or len(gt)!=2 or None in gt or not any(gt):return None,None
    if any(g<0 or g>=len(rec.alleles) for g in gt):raise ValueError('Invalid GT')
    seq=[rec.ref]+[rec.alleles[g] for g in gt]
    if any(not a or set(a)-set('ACGT') for a in seq):return None,None
    pos=rec.start
    while min(map(len,seq))>1 and len({a[-1] for a in seq})==1:seq=[a[:-1] for a in seq]
    while min(map(len,seq))>1 and len({a[0] for a in seq})==1:
        seq=[a[1:] for a in seq];pos+=1
    ref=seq[0];called=tuple(sorted(seq[1:]));alt={a for a in called if a!=ref}
    if not alt:return None,None
    if all(len(a)==len(ref)==1 for a in alt):kind=0
    elif all(0<abs(len(a)-len(ref))<=50 for a in alt):kind=1
    else:return None,None
    return (rec.chrom,pos,ref,called),kind

def balance(values, gt):
    v=vector(values)
    if not v or any(not math.isfinite(x) or x<0 for x in v) or max(gt)>=len(v):return float('nan'),float('nan')
    total=sum(v)
    if total<=0:return 0.,float('nan')
    nonref={g for g in gt if g>0};fraction=sum(v[g] for g in nonref)/total
    expected=sum(g>0 for g in gt)/2
    return math.log1p(total),abs(fraction-expected)

def posterior(value):
    p=vector(value)
    if not p or any(not math.isfinite(x) or x<0 for x in p) or sum(p)<=0:return [float('nan')]*4
    total=sum(p);p=[x/total for x in p];ranked=sorted(p,reverse=True)
    return [ranked[0],ranked[0]-(ranked[1] if len(ranked)>1 else 0),-sum(x*math.log(max(x,1e-30)) for x in p),p[0]]

def features(rec,sample,context_dp):
    c=rec.samples[sample];gt=c['GT'];dp=number(c.get('DP'));adp=number(c.get('ADP'))
    gp=posterior(c.get('GP'));ngp=posterior(c.get('NGP'));bgp=posterior(c.get('BGP'))
    ad=balance(c.get('AD'),gt);aad=balance(c.get('AAD'),gt)
    pl=sorted(x for x in vector(c.get('PL')) if math.isfinite(x))
    af=[x for x in vector(rec.info.get('PANEL_AF')) if math.isfinite(x)]
    called=[rec.alleles[g] for g in gt if g>0];size=max(abs(len(a)-len(rec.ref)) for a in called)
    source=rec.info.get('SOURCE','');effective=max([0.]+[x for x in [dp,adp] if math.isfinite(x)])
    ratio=effective/(context_dp+1) if math.isfinite(context_dp) else float('nan')
    return [lognum(dp),lognum(adp),lognum(context_dp),min(100.,ratio) if math.isfinite(ratio) else ratio,number(c.get('GQ')),lognum(rec.qual),
            *gp,ngp[0],bgp[0],*ad,*aad,min(999.,pl[1]-pl[0]) if len(pl)>1 else float('nan'),
            math.log1p(max(map(len,rec.alleles))),math.log1p(size),float(gt[0]!=gt[1]),
            min(af) if af else float('nan'),max(af) if af else float('nan'),
            float(source in ['PANEL','BOTH']),float(source in ['READS','BOTH']),float(bool(rec.info.get('REFINED',False))),
            lognum(rec.info.get('NEW_ALLELES',0)),number(c.get('V33PC')),number(c.get('V33PV')),
            float(rec.info.get('V36SRC')=='current')]

def depth_profile(path, width=50000):
    """Median DP of unique biallelic SNV records, not physical BAM coverage."""
    import pysam
    hist=collections.defaultdict(collections.Counter);seen=None
    with pysam.VariantFile(path) as f:
        if len(f.header.samples)!=1:raise ValueError('Single sample only')
        sample=list(f.header.samples)[0]
        for r in f:
            if len(r.alleles)!=2 or any(len(a)!=1 or a not in 'ACGT' for a in r.alleles):continue
            position=(r.chrom,r.start)
            if position==seen:continue
            seen=position;dp=r.samples[sample].get('DP')
            if dp is not None and dp>=0:hist[r.chrom,r.start//width][min(dp,10000)]+=1
    med={}
    for key,h in hist.items():
        n=sum(h.values())
        if n<20:continue
        cumulative=0
        for dp,count in sorted(h.items()):
            cumulative+=count
            if cumulative>=(n+1)//2:med[key]=float(dp);break
    return med

def infer(model, x):
    """Portable histogram-boosting inference from reviewed JSON trees."""
    # sklearn compares histogram split thresholds in float64. In float32,
    # NumPy can round a midpoint threshold onto an adjacent observed value
    # and send that value down the wrong branch.
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(NAMES):raise ValueError('Feature schema mismatch')
    logits=np.full(len(x),model['baseline'],dtype=np.float64)
    for tree in model['trees']:
        pending=[(0,np.arange(len(x)))]
        while pending:
            idx,rows=pending.pop()
            if not len(rows):continue
            n=tree[idx]
            if n['leaf']:logits[rows]+=n['value'];continue
            v=x[rows,n['feature']];left=np.where(np.isnan(v),n['missing_left'],v<=n['threshold'])
            pending.append((n['left'],rows[left]));pending.append((n['right'],rows[~left]))
    return 1/(1+np.exp(-np.clip(logits,-40,40)))
