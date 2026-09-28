"""Genotype-specific read count and alternate-sequence evidence; no truth input."""
import collections,math
import numpy as np
from learning import pairs,max_run
NAMES=['AD_LL_01','AD_LL_10','AD_LL_gain_01','AD_LL_gain_10','AAD_LL_02','AD_balance_error','AD_other_fraction','AD_minor_log','AD_total_log','support_missing','alt_trim_ref_log','alt_trim_alt_log','alt_entropy','alt_gc_change','alt_homopolymer','alt_period2','alt_period3','alt_trim_mismatch_fraction','alt_complex_indel','alt_MNP','alt_max_length_log','alt_nonref_copies']
def allele_properties(ref,alt):
 if ref==alt:return np.zeros(11,dtype=np.float32)
 left=0
 while left<min(len(ref),len(alt)) and ref[left]==alt[left]:left+=1
 right=0
 while right<min(len(ref)-left,len(alt)-left) and ref[len(ref)-1-right]==alt[len(alt)-1-right]:right+=1
 rr=ref[left:len(ref)-right if right else len(ref)];aa=alt[left:len(alt)-right if right else len(alt)]
 counts=collections.Counter(alt);entropy=-sum(n/len(alt)*math.log2(n/len(alt)) for n in counts.values())
 mismatch=sum(a!=b for a,b in zip(rr,aa))/max(1,min(len(rr),len(aa)))
 gc=lambda s:sum(b in 'GC' for b in s)/max(1,len(s))
 return np.array([math.log1p(len(rr)),math.log1p(len(aa)),entropy,gc(alt)-gc(ref),max_run(alt,1),max_run(alt,2),max_run(alt,3),mismatch,float(len(rr)!=len(aa) and bool(rr) and bool(aa)),float(len(rr)==len(aa) and len(rr)>1),math.log1p(len(alt))],np.float32)
def extra(rec,sample,choices):
 c=rec.samples[sample];order=pairs(len(rec.alleles));n=len(rec.alleles);original=tuple(c['GT']);properties=np.stack([allele_properties(rec.ref,a) for a in rec.alleles]);rows=[]
 def counts(key):
  v=c.get(key)
  return np.asarray(v,float) if v and len(v)==n and all(x is not None and math.isfinite(x) and x>=0 for x in v) and sum(v)>0 else None
 ad,aad=counts('AD'),counts('AAD')
 def ll(v,gt,error):
  if v is None:return -20.
  copies=np.bincount(gt,minlength=n)/2;p=copies*(1-error)+error/n
  return float(v@np.log(np.maximum(p,1e-12))/v.sum())
 old=[ll(ad,original,e) for e in [.01,.1]]
 for j in choices:
  gt=order[int(j)];loglike=[ll(ad,gt,e) for e in [.01,.1]]
  if ad is None:balance=other=minor=total=-1.
  else:
   total=float(ad.sum());selected=sum(ad[k] for k in set(gt));other=1-selected/total;minor=math.log1p(min(ad[k] for k in set(gt)));balance=abs(ad[gt[0]]/max(1,selected)-.5) if gt[0]!=gt[1] else other;total=math.log1p(total)
  p=properties[list(gt)].max(axis=0)
  # GC change is signed: max with reference zero erases negative changes.
  p[3]=properties[list(gt),3].mean()
  rows.append([*loglike,loglike[0]-old[0],loglike[1]-old[1],ll(aad,gt,.02),balance,other,minor,total,float(ad is None),*p,float(sum(g>0 for g in gt))])
 result=np.asarray(rows,np.float32)
 if result.shape!=(len(choices),len(NAMES)) or not np.isfinite(result).all():raise ValueError('Invalid additional evidence')
 return result
