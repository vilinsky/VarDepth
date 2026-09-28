"""Preserve the complete r05 candidate catalog, append independently assembled proposals."""
from baseline_proposals import propose as baseline_propose
from assembly import propose as assemble_propose

def merge_catalogs(before,assembled,fetch,maximum=36):
 start=min(before['start0'],assembled['start0']);end=max(before['end0'],assembled['end0'])
 def pad(row):return [fetch(start,row['start0'])+a+fetch(row['end0'],end) for a in row['alleles']]
 old=pad(before);new=pad(assembled);seen=set(old);added=[]
 for allele in new:
  if allele not in seen and len(old)+len(added)<maximum:
   added.append(allele);seen.add(allele)
 return dict(before,start0=start,end0=end,alleles=old+added,added_alleles=before['added_alleles']+len(added)),len(added)

def propose(row,original,bams,fetch,maximum=36):
 before,old_summary=baseline_propose(row,original,bams,fetch,maximum=24)
 assembled,assembly_summary=assemble_propose(row,original,bams,fetch,maximum=36)
 merged,n=merge_catalogs(before,assembled,fetch,maximum)
 summary=dict(added_beyond_r05=n,preserved_r05_alleles=len(before['alleles']),baseline=old_summary,assembly=assembly_summary)
 merged['preserved_assembly']=summary
 return merged,summary
