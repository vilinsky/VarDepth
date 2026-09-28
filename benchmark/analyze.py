"""Paired genomic block bootstrap of allowed aggregate counts, no individual calls."""
import datetime,hashlib,json,sys,time
from pathlib import Path
import numpy as np
O=Path('outputs/paper-check-20260929');COUNTS=['query_TP','truth_TP','FP','FN']
CASES=['lowpass','wes','hybrid','wgs'];METHODS=['vardepth44','vardepth43','vardepth42','gatk','deepvariant','glimpse2','quilt2']
KINDS=['SNP','INDEL_1_50'];CONTEXTS=['all','capture','outside'];B=10000;SEED=20260929
WGS=[1,2,3,4,6,8,10,12,14,16,18,20]
def save(name,d):(O/name).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def folder(m,c,n):
 if m in ['gatk','glimpse2']:
  worker='worker04' if m=='gatk' else 'worker03'
  return Path('outputs/development-v41')/worker/'collected/assessment'/m/c/f'chr{n}'/'PASS'
 return O/'counts'/m/c/f'chr{n}'
def metrics(v):
 v=np.asarray(v,dtype=float);q,t,fp,fn=np.moveaxis(v,-1,0)
 with np.errstate(divide='ignore',invalid='ignore'):
  p=q/(q+fp);r=t/(t+fn);f=2*p*r/(p+r)
  f=np.where((p==0)&(r==0),0,f)
 return np.stack([p,r,f],axis=-1)
def interval(x):
 valid=np.isfinite(x);a=x[valid]
 return dict(low=float(np.quantile(a,.025)) if len(a) else None,high=float(np.quantile(a,.975)) if len(a) else None,valid_repetitions=int(valid.sum()))
def load_all():
 arrays={};manifest=[];checks=0
 for m in METHODS:
  for case in CASES:
   if m=='quilt2' and case=='wgs':continue
   for n in (WGS if case=='wgs' else range(1,23)):
    p=folder(m,case,n);meta=json.loads((p/'COUNTS.json').read_text())
    assert meta['success'] and meta['official_aggregate_parity'] and meta['block_bp']==1000000
    assert meta['count_order']==COUNTS
    for ctx in CONTEXTS[1:]:assert ctx in meta['official_stratum_parity']
    with np.load(p/'counts.npz',allow_pickle=False) as d:
     assert d.files==['counts'];a=d['counts']
    assert a.ndim==4 and a.shape[:2]==(7,10) and a.shape[-1]==4 and a.dtype.kind in 'iu' and (a>=0).all()
    assert np.array_equal(a[:,1],a[:,2:].sum(axis=1))
    for r in meta['rows']:
     ci=meta['contexts'].index(r['context']);k=r['kind']
     v=a[ci,2:7].sum(axis=(0,1)) if k=='INDEL_1_50' else a[ci,meta['labels'].index(k)].sum(axis=0)
     assert np.array_equal(v,[r[x] for x in COUNTS]),(m,case,n,k,r['context'])
     actual=metrics(v)
     for x,y in zip(actual,[r['precision'],r['recall'],r['f1']]):
      if y is not None and np.isfinite(x):assert abs(x-y)<1e-12
     checks+=1
    arrays[m,case,n]=(a,meta)
    if len(arrays)%25==0:print("Verified aggregate sets",len(arrays),flush=True)
    for name in ['COUNTS.json','counts.npz']:
     raw=(p/name).read_bytes();manifest.append(dict(path=str(p/name),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
 save('SOURCE_PARITY.json',dict(success=True,chromosome_method_case_sets=len(arrays),metadata_rows_verified=checks,files=manifest,block_tensors_only=True))
 return arrays

def calculate(arrays):
 rows=[];comparisons=[];block_info=[]
 # Cache common paired weights: identical selection for every method/assay.
 weights={}
 for width in [2,1,4]:
  rng=np.random.default_rng(SEED+width)
  for n in range(1,23):
   size=arrays['vardepth44','lowpass',n][0].shape[2];blocks=(size+width-1)//width
   weights[width,n]=rng.multinomial(blocks,np.full(blocks,1/blocks),size=B)
 for scope,widths in [('primary_chr18_chr20',[2,1,4]),('secondary_capture',[2])]:
  for width in widths:
   for case in CASES:
    ns=[18,20] if scope.startswith('primary') else (WGS if case=='wgs' else list(range(1,23)))
    for ctx in (CONTEXTS if scope.startswith('primary') else ['capture']):
     for kind in KINDS:
      distributions={};point={}
      for m in METHODS:
       if m=='quilt2' and (case=='wgs' or kind!='SNP'):continue
       total=np.zeros(4,np.int64);rep=np.zeros((B,4),np.int64);nonempty=0;numblocks=0
       for n in ns:
        a,meta=arrays[m,case,n];ci=meta['contexts'].index(ctx)
        counts=a[ci,0] if kind=='SNP' else a[ci,2:7].sum(axis=0)
        coarse=np.add.reduceat(counts,np.arange(0,len(counts),width),axis=0)
        w=weights[width,n];assert w.shape[1]==len(coarse)
        total+=coarse.sum(axis=0);rep+=w@coarse
        nonempty+=int((coarse.sum(axis=1)>0).sum());numblocks+=len(coarse)
       values=metrics(total);sampled=metrics(rep);distributions[m]=sampled[:,2];point[m]=float(values[2])
       row=dict(scope=scope,block_bp=width*1000000,method=m,case=case,context=ctx,kind=kind,chromosomes=ns,blocks=numblocks,nonempty_blocks=nonempty,**dict(zip(COUNTS,map(int,total))),**dict(zip(['precision','recall','f1'],map(float,values))),intervals={key:interval(sampled[:,i]) for i,key in enumerate(['precision','recall','f1'])})
       rows.append(row)
      for m in distributions:
       if m=='vardepth44':continue
       delta=distributions['vardepth44']-distributions[m];ci=interval(delta)
       comparisons.append(dict(scope=scope,block_bp=width*1000000,case=case,context=ctx,kind=kind,comparator=m,delta_f1=point['vardepth44']-point[m],interval=ci,regional_direction=('positive' if ci['low']>0 else 'negative' if ci['high']<0 else 'unresolved')))
 # Explicit parity to the already published compact aggregation; no new scores.
 old=json.loads(Path('outputs/paper-compact/MATCHED_COMPARISON.json').read_text())
 lookup={(r['method'],r['case'],r['context'],r['kind']):r for r in rows if r['scope']=='primary_chr18_chr20' and r['block_bp']==2000000}
 for r in old['rows']:
  new=lookup[tuple(r[x] for x in ['method','case','context','kind'])]
  assert all(new[k]==r[k] for k in COUNTS) and abs(new['f1']-r['f1'])<1e-12
 assert len(lookup)==153
 result=dict(success=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sample='HG005',independent_person_validation=False,previously_exposed_regions=True,bootstrap_repetitions=B,seed=SEED,seed_rule='seed + block width in Mb; chromosomes numeric order; same paired weights for all methods and cases',bootstrap_method='Within each chromosome resample fixed genomic blocks with replacement, same block index multiplicities for every method; sum counts then recompute asymmetric-TP metrics',block_anchor='existing first benchmark BS superlocus anchor, otherwise POS',confidence_interpretation='Pointwise conditional regional percentile intervals; ignores across-person variation and correlations beyond chosen blocks; zero observed errors cannot establish zero true error; no multiple-comparison adjusted superiority claim',prior_compact_aggregate_rows_reproduced=153,rows=rows,comparisons=comparisons)
 save('ANALYSIS.json',result)
 print(json.dumps(dict(success=True,rows=len(rows),comparisons=len(comparisons),parity=153)))
 for c in comparisons:
  if c['scope']=='primary_chr18_chr20' and c['block_bp']==2000000 and ((c['case'] in ['lowpass','hybrid'] and c['context']=='outside' and c['comparator'] in ['glimpse2','quilt2']) or (c['case']=='wgs' and c['context']=='all' and c['comparator'] in ['gatk','deepvariant'])):
   print(c['case'],c['kind'],c['comparator'],round(c['delta_f1']*100,4),[round(c['interval'][x]*100,4) for x in ['low','high']])
if __name__=='__main__':calculate(load_all())
