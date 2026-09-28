"""Experimental removal of duplicated indel events across haplotype blocks.

A posterior-majority retained source owns an identical called event. The child retains
all its other edits, with exact summation of its original GP under the allele
projection. Dependent block probabilities are never multiplied. This is source
selection and posterior projection, not a new calibrated joint posterior.
"""
import argparse,collections,hashlib,json,math
from pathlib import Path
import numpy as np
import pysam
from ownership_events import catalog_events,apply,marginals

def order(n):return [(a,b) for b in range(n) for a in range(b+1)]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def projected_catalog(start,alleles,catalog,remove,fetch):
 left=min([start]+[p for events in catalog for p,r,a in events]);right=max([start+len(alleles[0])]+[p+len(r) for events in catalog for p,r,a in events])
 reference=fetch(left,right);new=[];mapping=[]
 for events in catalog:
  sequence=apply(reference,left,events-set(remove))
  if sequence not in new:new.append(sequence)
  mapping.append(new.index(sequence))
 if len(new)>1:
  while min(map(len,new))>1 and len({a[-1] for a in new})==1:new=[a[:-1] for a in new]
  while min(map(len,new))>1 and len({a[0] for a in new})==1:left+=1;new=[a[1:] for a in new]
 return left,new,mapping
def project_gp(gp,mapping):
 p=np.asarray(gp,dtype=float);n=len(mapping);outn=max(mapping)+1
 if p.shape!=(n*(n+1)//2,) or not np.isfinite(p).all() or min(p)<0 or p.sum()<=0:raise ValueError('Invalid GP')
 out=np.zeros(outn*(outn+1)//2)
 for mass,(a,b) in zip(p/p.sum(),order(n)):
  x,y=sorted((mapping[a],mapping[b]));out[y*(y+1)//2+x]+=mass
 return out
def called_pairs(catalog,record,samples):
 result={}
 for sample in samples:
  gt=record.samples[sample].get('GT')
  if gt is None or len(gt)!=2 or None in gt:raise ValueError('Diploid called genotype required')
  result[sample]=tuple(catalog[g] for g in gt)
 return result
def same_phase(parent,child,events):
 for sample in child:
  p=tuple(h&events for h in parent[sample]);c=tuple(h&events for h in child[sample])
  if p!=c and p[::-1]!=c:return False
 return True

def reconcile(input_path,reference_path,output_path,*,min_owner_probability=.5):
 if not .5<=min_owner_probability<=1:raise ValueError('Owner threshold must be a posterior majority')
 dest=Path(output_path)
 if dest.exists() or dest.resolve()==Path(input_path).resolve():raise ValueError('Fresh separate output required')
 dest.parent.mkdir(parents=True,exist_ok=True)
 original_sha=sha(input_path);descriptors={};totals=collections.Counter();proofs=[]
 with pysam.FastaFile(str(reference_path)) as fa,pysam.VariantFile(str(input_path)) as source:
  samples=list(source.header.samples);header=source.header.copy()
  header.info.add('EVENT_OWNERSHIP',0,'Flag','Experimental exact GP projection after duplicate indel source selection; original posterior VCF retained')
  for index,r in enumerate(source):
   totals['input_records']+=1
   if not any(len(a)!=len(r.ref) for a in r.alts or []):continue
   if max(int(r.samples[s].get('DP',0) or 0) for s in samples)<20:continue
   fetch=lambda s,e:fa.fetch(r.chrom,max(0,s),e).upper()
   try:
    catalog=catalog_events(r.start,[a.upper() for a in r.alleles],fetch);pairs=called_pairs(catalog,r,samples)
    used=set().union(*(h for pair in pairs.values() for h in pair));used={e for e in used if len(e[1])!=len(e[2])}
    if not used:continue
    probabilities={s:marginals(catalog,r.samples[s]['GP'],r.samples[s]['GT']) for s in samples}
    confidence={e:min(probabilities[s][e]['gp'][sum(probabilities[s][e]['gt'])] for s in samples) for e in used}
   except (ValueError,KeyError,IndexError):totals['ineligible_catalog_or_gp']+=1;continue
   descriptors[index]=dict(chrom=r.chrom,start=r.start,end=r.stop,catalog=catalog,pairs=pairs,used=used,confidence=confidence,
    rank=(min(confidence.values()),len(r.ref),len(used),-index),alleles=list(r.alleles))
  owners=collections.defaultdict(set);removed={}
  for index in sorted(descriptors,key=lambda i:descriptors[i]['rank'],reverse=True):
   child=descriptors[index];possible=collections.Counter()
   for e in child['used']:
    for parent in owners[(child['chrom'],e)]:possible[parent]+=1
   for parent in sorted(possible,key=lambda j:(possible[j],descriptors[j]['rank']),reverse=True):
    p=descriptors[parent]
    events=frozenset(e for e in child['used'] if parent in owners[(child['chrom'],e)])
    if not same_phase(p['pairs'],child['pairs'],events):continue
    removed[index]=events
    proofs.append(dict(child_index=index,owner_index=parent,events=sorted(events),owner_confidence={str(e):p['confidence'][e] for e in events},proof='Identical called canonical indel events and diploid phase up to chromosome swap in every sample; source dosage marginal exceeds the registered majority threshold',owner_probability_threshold=min_owner_probability))
    break
   for e in child['used']-set(removed.get(index,[])):
    if child['confidence'][e]>=min_owner_probability:owners[(child['chrom'],e)].add(index)
  records=[]
  with pysam.VariantFile(str(input_path)) as original:
   for index,r in enumerate(original):
    if index not in removed:
     r.translate(header);records.append(r);continue
    d=descriptors[index];fetch=lambda s,e:fa.fetch(r.chrom,max(0,s),e).upper()
    start,alleles,mapping=projected_catalog(r.start,[a.upper() for a in r.alleles],d['catalog'],removed[index],fetch)
    totals['projected_records']+=1;totals['removed_duplicate_events']+=len(removed[index])
    if len(alleles)==1:totals['empty_catalogs_removed']+=1;continue
    v=header.new_record(contig=r.chrom,start=start,stop=start+len(alleles[0]),alleles=alleles)
    v.info['EVENT_OWNERSHIP']=True
    if 'SOURCE' in r.info:v.info['SOURCE']=r.info['SOURCE']
    passing=False;logref=0.
    for s in samples:
     call=r.samples[s];out=v.samples[s];p=project_gp(call['GP'],mapping);gt=tuple(sorted(mapping[g] for g in call['GT']));a,b=gt;g=b*(b+1)//2+a
     out['GT']=gt;out['GP']=tuple(p);out['GQ']=min(99,round(-10*math.log10(max(1e-12,1-p[g]))))
     for field in ['DP','CS']:
      if call.get(field) is not None:out[field]=call[field]
     out['DS']=tuple(sum(mass*(int(a==j)+int(b==j)) for mass,(a,b) in zip(p,order(len(alleles)))) for j in range(1,len(alleles)))
     passing|=any(gt) and out['GQ']>=20;logref+=math.log(max(1e-100,p[0]))
    v.qual=min(999.,-10/math.log(10)*logref);v.filter.add('PASS' if passing else 'LowQual');records.append(v)
  contigs={c:i for i,c in enumerate(header.contigs)};records.sort(key=lambda r:(contigs[r.chrom],r.start,r.alleles))
  with pysam.VariantFile(str(dest),'wz',header=header) as out:
   for r in records:out.write(r)
 pysam.tabix_index(str(dest),preset='vcf',force=True)
 if sha(input_path)!=original_sha:raise ValueError('Input changed')
 totals['output_records']=len(records)
 result=dict(counts=dict(totals),input_sha256=original_sha,output_sha256=sha(dest),proofs=proofs,truth_used=False,
  owner_probability_threshold=min_owner_probability,source_ranking='minimum called-event dosage confidence, then REF span and event count',
  scope='Experimental called-event source selection. GP exactly projected within each source, GT inherited from original joint MAP; likelihood/AD/frequency vectors omitted at projected records because they cannot be relabelled as posteriors. Not calibrated joint fusion across blocks.')
 Path(str(dest)+'.event-ownership.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 print(json.dumps(reconcile(a.input,a.reference,a.output)['counts']))
