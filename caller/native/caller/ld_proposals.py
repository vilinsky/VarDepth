"""One bounded LD-first local sequence-proposal pass, without posterior reuse."""
from __future__ import annotations
import collections


def select_proposals(initial, blocks):
    ids_by_first={b['ids'][0]:tuple(b['ids']) for b in blocks if b['kind']=='multiallelic'}
    weights=collections.defaultdict(collections.Counter)
    for sample,result in initial.items():
        for first,proposals in result[2].get('ld_sequence_proposals',{}).items():
            first=int(first)
            if first not in ids_by_first:continue
            ids=ids_by_first[first]
            for proposal in proposals:
                vector=tuple(proposal['vector'])
                if len(vector)!=len(ids) or any(x not in (0,1) for x in vector):raise ValueError('Invalid LD proposal encoding')
                # Maximum per-person occupancy avoids multiplying the same
                # observed haplotype just because more people share it.
                weights[ids][vector]=max(weights[ids][vector],float(proposal['weight']))
    return {ids:[v for v,w in sorted(values.items(),key=lambda x:(-x[1],x[0]))[:2]] for ids,values in weights.items()}


def changed_blocks(before,after):
    if len(before)!=len(after) or any(a['ids']!=b['ids'] for a,b in zip(before,after)):
        raise ValueError('LD proposal stage must preserve shared seed-block identities')
    return [i for i,(a,b) in enumerate(zip(before,after)) if a.get('alleles')!=b.get('alleles')]


def preserve_read_supported_alleles(before,after,variants,read_sources):
    """Never remove an existing local sequence representing a supported seed.

    The original read-path quota is already retained. This additional guard
    prevents a promoted panel path displacing a uniquely represented supported
    alternate among the remaining population-ranked slots.
    """
    for old,new in zip(before,after):
        if old.get('alleles')==new.get('alleles') or old['kind']!='multiallelic':continue
        removed=set(old['alleles'])-set(new['alleles'])
        protected={seq for seq,reps in zip(old['alleles'],old['representations'])
                   if any(any(v[j] and read_sources.get(variants[index],0) for j,index in enumerate(old['ids'])) for v in reps)}
        if removed & protected:
            new.clear();new.update(old)
    return after
