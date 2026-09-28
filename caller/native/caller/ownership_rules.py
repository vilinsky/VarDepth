"""Suppress redundant confident VCF records, retaining original source posteriors.

Canonical edits are used only to prove that the called diploid haplotypes of
one record are already described by another record. Output is never atomized.
No posteriors of dependent records are multiplied or averaged. This is source
selection, not posterior marginalization over the union of candidate catalogs.
"""
import argparse,collections,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import pysam
from ownership_events import catalog_events,minimal
THRESHOLD=.99

def sha(p):
 digest=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):digest.update(block)
 return digest.hexdigest()
def span(event):
 p,r,a=minimal(*event);return p,p+len(r)
def touches(event,left,right):
 a,b=span(event);return left<=a<=right if a==b else a<right and b>left

def implied(owner,child,left,right):
 """Preserve cis/trans structure up to the arbitrary diploid chromosome swap."""
 projected=[]
 for hap in owner:
  selected=set()
  for e in hap:
   if not touches(e,left,right):continue
   a,b=span(e)
   if a<left or b>right:return False
   selected.add(e)
  projected.append(frozenset(selected))
 return tuple(projected)==tuple(child) or tuple(projected[::-1])==tuple(child)

def describe(record,samples,fetch):
 if any(not a or not set(a.upper())<=set('ACGT') for a in record.alleles):return None
 used={0};gts={};confidence=[]
 for sample in samples:
  call=record.samples[sample];gt=call.get('GT');gp=call.get('GP');n=len(record.alleles)
  if gt is None or len(gt)!=2 or None in gt or gp is None or len(gp)!=n*(n+1)//2:return None
  a,b=sorted(gt);prob=float(gp[b*(b+1)//2+a]);confidence.append(prob)
  if not math.isfinite(prob) or prob<THRESHOLD:return None
  used.update(gt);gts[sample]=gt
 if used=={0}:return None
 indices=sorted(used)
 try:catalog=catalog_events(record.start,[record.alleles[i].upper() for i in indices],fetch)
 except ValueError:return None
 mapping=dict(zip(indices,catalog));pairs={s:tuple(mapping[i] for i in gt) for s,gt in gts.items()}
 union=frozenset().union(*(h for pair in pairs.values() for h in pair))
 if not union:return None
 # Include reference context and any equivalent left-shifted canonical edits.
 left=min([record.start]+[span(e)[0] for e in union]);right=max([record.stop]+[span(e)[1] for e in union])
 return dict(pairs=pairs,events=union,left=left,right=right,confidence=min(confidence),rank=(len(record.ref),len(union),min(confidence)))

def redundant_records(descriptions):
 owners=collections.defaultdict(set);removed={}
 for index in sorted(descriptions,key=lambda i:(descriptions[i]['rank'],-i),reverse=True):
  child=descriptions[index];possible=None
  for event in child['events']:
   candidates=owners[event]
   possible=set(candidates) if possible is None else possible & candidates
   if not possible:break
  valid=[]
  for candidate in possible or []:
   parent=descriptions[candidate]
   if all(implied(parent['pairs'][s],child['pairs'][s],child['left'],child['right']) for s in child['pairs']):valid.append(candidate)
  if valid:
   owner=max(valid,key=lambda i:(descriptions[i]['rank'],-i));removed[index]=owner
  else:
   for event in child['events']:owners[event].add(index)
 return removed
