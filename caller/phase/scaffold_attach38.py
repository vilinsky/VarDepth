"""Conservative, one-pass attachment to a frozen phase scaffold.

No truth inputs. A molecule contributes once per target/block, irrespective of
how many scaffold SNPs it overlaps. New attachments never become anchors.
Scores are model quantities; they are not clinical calibration guarantees.
"""
import collections
import math


def attach(assignments, molecules, max_error=.01, minimum_support=3, forbidden=()):
    frozen=dict(assignments)
    forbidden=set(forbidden)
    tables={};stats=collections.Counter()
    for observations in molecules:
        anchors=collections.defaultdict(list)
        for node,value in observations.items():
            if value is not None and node in frozen:
                root,bit,bound=frozen[node]
                allele,error=value
                anchors[root].append((allele ^ bit,error,node,bound))
        for node,value in observations.items():
            if value is None or node in frozen or node in forbidden:continue
            allele,error=value
            for root,group in anchors.items():
                # Conflicting SNPs on the same physical molecule are missing
                # evidence, not votes that can be averaged into confidence.
                if len({g[0] for g in group})!=1:
                    stats['discordant_molecule_blocks']+=1;continue
                hap,anchor_error,anchor,bound=min(group,key=lambda g:(g[1],g[2]))
                e=error+anchor_error-2*error*anchor_error
                if not 0<e<.5:continue
                bit=allele ^ hap
                row=tables.setdefault((node,root),dict(score=0.,n=0,votes=[0,0],alleles=[0,0],bound=bound))
                row['score']+=(1 if bit==0 else -1)*math.log((1-e)/e)
                row['n']+=1;row['votes'][bit]+=1;row['alleles'][allele]+=1
    candidates=collections.defaultdict(list)
    for (node,root),row in tables.items():
        if row['n']<minimum_support:
            stats['insufficient_molecules']+=1;continue
        if min(row['alleles'])==0:
            stats['one_allele_only']+=1;continue
        bit=int(row['score']<0)
        if row['votes'][bit]/row['n']<.9:
            stats['conflicting_support']+=1;continue
        error=max(.0001,1/(1+math.exp(min(700,abs(row['score'])))))
        # Two new leaves may be queried together: reserve both error terms.
        if row['bound']+2*error>max_error:
            stats['pair_path_budget']+=1;continue
        candidates[node].append((error,-row['n'],root,bit,row))
    selected=[];block_extra=collections.defaultdict(float)
    for node,choices in sorted(candidates.items()):
        error,negative_n,root,bit,row=min(choices,key=lambda x:x[:4])
        selected.append(dict(node=node,root=root,bit=bit,error=error,molecules=-negative_n,
                             allele_support=row['alleles'],parity_support=row['votes']))
        block_extra[root]=max(block_extra[root],error)
    out={node:(root,bit,bound+2*block_extra.get(root,0.)) for node,(root,bit,bound) in frozen.items()}
    for row in selected:
        root=row['root'];bound=frozen[root][2]+2*block_extra[root]
        out[row['node']]=(root,row['bit'],bound)
    stats.update(attachments=len(selected),blocks_extended=len(block_extra),molecule_block_tables=len(tables))
    return out,dict(stats),selected


def overlapping_indices(records):
    """Quarantine all overlapping records, including long containing alleles."""
    active=[];bad=set()
    for i,record in enumerate(records):
        active=[j for j in active if records[j].stop>record.start]
        if active:bad.update(active);bad.add(i)
        active.append(i)
    return bad
