"""Two rounds of partial-window sequence proposals; no inferred flank evidence.

Candidate generation uses read sequences alone. Read likelihoods must later be
recomputed; the score here is only a bounded proposal-ranking heuristic.
"""
import collections
import numpy as np
from projection import common_anchors,crop_observation,expand_compound_context,project

def splice(template,mapping,left,right,sequence):
 a,b=int(mapping[left]),int(mapping[right-1])+1
 if a<0 or b<=a or b>len(template):return None
 return template[:a]+sequence+template[b:]

def assemble(reference,templates,observations,valid,maximum=36,rounds=2,beam=48):
 """Keep the original ordered prefix; merge visible windows on a bounded beam.

 Observations are (physical molecule ID, left, right, sequence), in reference
 coordinates relative to the padded context. Duplicate mates share an ID.
 """
 original=list(dict.fromkeys(templates));retained=list(original);seen=set(original)
 maps={s:project(reference,s)[0] for s in original};support={}
 def score(sequence):
  mapping=maps.get(sequence)
  if mapping is None:
   mapping=project(reference,sequence)[0];maps[sequence]=mapping
  molecules={}
  for key,left,right,observed in observations:
   a,b=int(mapping[left]),int(mapping[right-1])+1
   if a<0 or b<=a:continue
   # Reference-identical windows do not inflate alternate support.
   if observed==reference[left:right] or sequence[a:b]!=observed:continue
   molecules[key]=max(molecules.get(key,0),right-left)
  return molecules
 def ranking(sequence):return (-len(support[sequence]),-sum(support[sequence].values()),sequence)
 for iteration in range(rounds):
  proposals=set()
  for template in retained:
   mapping=maps[template]
   for key,left,right,observed in observations:
    sequence=splice(template,mapping,left,right,observed)
    if sequence is None or sequence in seen or not valid(sequence):continue
    proposals.add(sequence)
  for sequence in sorted(proposals):
   maps[sequence]=project(reference,sequence)[0]
   support[sequence]=score(sequence);seen.add(sequence)
  ranked=sorted((s for s in support if len(support[s])>=2),key=ranking)
  retained=original+ranked[:max(0,beam-len(original))]
 ranked=sorted((s for s in support if len(support[s])>=2),key=ranking)
 selected=ranked[:max(0,min(24,maximum-len(original)))]
 return original+selected,dict(eligible=len(ranked),examined=len(seen),rounds=rounds,
  proposals={s:dict(distinct_molecules=len(support[s]),summed_observed_span=sum(support[s].values())) for s in selected})

def propose(row,original,bams,fetch,maximum=36):
 start,old=expand_compound_context(row['start0'],row['alleles'],fetch);end=start+len(old[0])
 prefix=fetch(start-3,start);suffix=fetch(end,end+3)
 if len(prefix)!=3 or len(suffix)!=3 or len(old)>=maximum:return row,dict(added=0)
 reference=prefix+old[0]+suffix;templates=[prefix+a+suffix for a in old]
 maps,safe=common_anchors(start-3,reference,templates,fetch);observations=[]
 for sample,inputs in bams.items():
  for bam,rg,fallback in inputs:
   for read in bam.fetch(row['chrom'],max(0,start-3),end+3):
    if read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate or read.is_qcfail:continue
    if not 20<=read.mapping_quality<255 or read.query_qualities is None:continue
    obs=crop_observation(read,start-3,reference,templates,maps,safe)
    if obs is None or (sum(obs['qualities'])/len(obs['qualities']))<25 or any(b not in 'ACGT' for b in obs['sequence']):continue
    if min(obs['qualities'][:3]+obs['qualities'][-3:])<20:continue
    library=rg.get(read.get_tag('RG'),fallback) if read.has_tag('RG') else fallback
    observations.append(((sample,library,read.query_name),obs['left']-(start-3),obs['right']-(start-3),obs['sequence']))
 observations=sorted(set(observations));lf=original['start0']-start;rf=end-original['end0']
 def valid(sequence):
  inner=sequence[3:-3]
  return (sequence.startswith(prefix) and sequence.endswith(suffix) and bool(inner) and abs(len(inner)-len(old[0]))<=50
          and (not lf or inner[:lf]==old[0][:lf]) and (not rf or inner[-rf:]==old[0][-rf:]))
 alleles,stats=assemble(reference,templates,observations,valid,maximum=maximum)
 new=[s[3:-3] for s in alleles];added=len(new)-len(old)
 summary=dict(stats,added=added,observations=len(observations),inference_truth_used=False)
 return dict(row,start0=start,end0=end,alleles=new,added_alleles=row['added_alleles']+added,assembled_proposals=summary),summary
