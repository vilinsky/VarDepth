"""CPU-only, position-free features and bounded genotype alternatives.

Inference accepts VCF and reference sequence only. Truth is confined to the
training driver. Existing read/panel/neural evidence is not multiplied as if
independent.
"""
import math,collections
import numpy as np
from legacy_features import features as old_features,NAMES as OLD_NAMES
EXTRA=['flank_GC','flank_entropy','flank_homopolymer','flank_period2_run','flank_period3_run','allele_balance_deviation','unassigned_AD_fraction','DP_AD_disagreement','GP_margin','NGP_margin','BGP_margin','GP_NGP_disagreement','GP_BGP_disagreement','GP_PL_disagreement','GP_NGP_TV','GP_BGP_TV']
CALL_NAMES=OLD_NAMES+EXTRA
RANK_NAMES=['original_'+n for n in CALL_NAMES]+['candidate_'+n for n in OLD_NAMES]+['is_original_GT','candidate_is_homref','log_GP_ratio_to_original','log_NGP_ratio_to_original','log_BGP_ratio_to_original']
def pairs(n):return [(a,b) for b in range(n) for a in range(b+1)]
def gt_index(gt):
 a,b=sorted(gt);return b*(b+1)//2+a
def probability(call,key,n):
 a=call.get(key)
 if a is None or len(a)!=n or any(v is None or not math.isfinite(v) or v<0 for v in a):return None
 x=np.asarray(a,float);s=x.sum();return x/s if s>0 else None
def eligible(r,sample):
 c=r.samples[sample];gt=c.get('GT')
 if not gt or len(gt)!=2 or None in gt or len(r.alleles)<2 or len(r.alleles)>48 or any(not a or set(a.upper())-set('ACGT') for a in r.alleles):return False
 if any(gt):return True
 n=len(r.alleles)*(len(r.alleles)+1)//2
 for key in ['GP','NGP','BGP']:
  p=probability(c,key,n)
  if p is not None and (p[0]<.99 or p.argmax()!=0):return True
 return False

def max_run(seq,period):
 best=current=0
 for i in range(period,len(seq)):
  current=current+1 if seq[i]==seq[i-period] else 0;best=max(best,current)
 return best+period if seq else 0

def call_features(r,sample,reference):
 x=old_features(r,sample);c=r.samples[sample];gt=c['GT'];n=len(r.alleles)*(len(r.alleles)+1)//2
 seq=reference[max(0,r.start-16):r.stop+16].upper();counts=collections.Counter(seq);length=max(1,len(seq))
 entropy=-sum((v/length)*math.log2(v/length) for v in counts.values());gc=sum(counts[b] for b in 'GC')/length
 ps=[probability(c,k,n) for k in ['GP','NGP','BGP']]
 margins=[float(np.sort(p)[-1]-np.sort(p)[-2]) if p is not None else -1 for p in ps]
 disagree=lambda a,b:float(a.argmax()!=b.argmax()) if a is not None and b is not None else -1.
 tv=lambda a,b:float(abs(a-b).sum()/2) if a is not None and b is not None else -1.
 pl=c.get('PL');plbest=int(np.argmin(pl)) if pl and len(pl)==n and all(p is not None and math.isfinite(p) for p in pl) else None
 ad=c.get('AD');dp=float(c.get('DP') or 0);balance=unassigned=dpdiff=-1.
 if ad and all(j<len(ad) and ad[j] is not None for j in gt):
  total=sum(v for v in ad if v is not None);selected=sum(ad[j] for j in set(gt));unassigned=1-selected/max(1,total);dpdiff=abs(dp-total)/max(1,dp,total)
  balance=abs(ad[gt[0]]/max(1,selected)-.5) if gt[0]!=gt[1] else 0.
 extra=[gc,entropy,max_run(seq,1),max_run(seq,2),max_run(seq,3),balance,unassigned,dpdiff,*margins,disagree(ps[0],ps[1]),disagree(ps[0],ps[2]),float(ps[0].argmax()!=plbest) if ps[0] is not None and plbest is not None else -1.,tv(ps[0],ps[1]),tv(ps[0],ps[2])]
 z=np.r_[x,np.asarray(extra,np.float32)].astype(np.float32)
 if len(z)!=len(CALL_NAMES) or not np.isfinite(z).all():raise ValueError('Invalid call features')
 return z

def proposals(r,sample):
 c=r.samples[sample];n=len(r.alleles)*(len(r.alleles)+1)//2;current=gt_index(c['GT']);selected=[current,0]
 for key in ['GP','NGP','BGP']:
  p=probability(c,key,n)
  if p is not None:selected.extend(np.argsort(-p,kind='stable')[:2].tolist())
 pl=c.get('PL')
 if pl and len(pl)==n and all(p is not None and math.isfinite(p) for p in pl):selected.append(int(np.argmin(pl)))
 return sorted(set(selected))

def rank_features(r,sample,reference,base=None):
 if base is None:base=call_features(r,sample,reference)
 original=tuple(r.samples[sample]['GT']);was_phased=r.samples[sample].phased;current=gt_index(original);order=pairs(len(r.alleles));choices=proposals(r,sample);n=len(order)
 ps=[probability(r.samples[sample],k,n) for k in ['GP','NGP','BGP']];result=[]
 try:
  for j in choices:
   r.samples[sample]['GT']=order[j]
   ratios=[float(np.clip(math.log(max(1e-12,p[j]))-math.log(max(1e-12,p[current])),-30,30)) if p is not None else 0 for p in ps]
   result.append(np.r_[base,old_features(r,sample),float(j==current),float(j==0),ratios])
 finally:r.samples[sample]['GT']=original;r.samples[sample].phased=was_phased
 return np.asarray(result,np.float32),np.asarray(choices,np.int32)

def kind(r,sample):
 gt=r.samples[sample]['GT'];return 'INDEL' if any(len(r.alleles[j])!=len(r.ref) for j in set(gt)) else 'SNV'
