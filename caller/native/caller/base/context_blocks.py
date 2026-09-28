"""Bounded nearby SNV/indel contexts, without truth or external caller output."""
import numpy as np
from local_alleles import overlap_groups, make_blocks


def context_groups(variants, gap=30, span=100):
    stack=[]
    for group in overlap_groups(variants):
        stack.append(group)
        while len(stack)>1:
            left,right=stack[-2:]
            end_left=max(variants[j][0]+len(variants[j][1]) for j in left)
            end_right=max(variants[j][0]+len(variants[j][1]) for j in right)
            has_indel=any(len(variants[j][1])!=len(variants[j][2]) for j in left+right)
            if not has_indel or variants[right[0]][0]-end_left>gap or end_right-variants[left[0]][0]>span:break
            stack[-2:]=[left+right]
    return stack


def make_context_blocks(variants,fetch,context_gap=30,context_span=100,partial_reads=True,**kwargs):
    groups=context_groups(variants,context_gap,context_span)
    blocks=make_blocks(variants,fetch,groups_override=groups,**kwargs)
    for t,block in enumerate(blocks):
        ids=block['ids'];start=variants[ids[0]][0]
        end=max(variants[j][0]+len(variants[j][1]) for j in ids)
        if block['kind']=='single' and len(variants[ids[0]][1])!=len(variants[ids[0]][2]):
            v=variants[ids[0]];entry=(kwargs.get('panel') or {}).get(v)
            block.update(kind='multiallelic',start=start,end=end,alleles=[v[1],v[2]],
                         representations=[[(0,)],[(1,)]],allele_af=[1-entry[1],entry[1]] if entry else [None,None])
        if block['kind']!='multiallelic':continue
        # No retained candidate outside this block occurs in its PairHMM flanks.
        # Read substrings are cut to this context; distant/unknown alternatives
        # cannot be integrated by the present bounded implementation.
        previous=max(variants[j][0]+len(variants[j][1]) for j in blocks[t-1]['ids']) if t else 0
        following=variants[blocks[t+1]['ids'][0]][0] if t+1<len(blocks) else end+81
        block['left']=max(0,start-80,previous)
        block['right']=min(end+81,following)
        block['context_span']=end-start
        block['partial_reads']=partial_reads
        block['marker_spans']=[(variants[j][0],variants[j][0]+len(variants[j][1])) for j in ids]
    return blocks
