"""Independently implemented bounded local de Bruijn haplotype proposals.

This module proposes sequences only. It assigns no genotype or probability.
Reference edges keep the reference route connected; every novel edge needs
support from distinct physical molecules. Repeated mates do not add support.
"""
import collections, math, re

CANONICAL_RUN = re.compile('[ACGT]+')

def graph(reference, observations, k=31, minimum=2):
    if k<3 or len(reference)<2*k:raise ValueError('Reference must contain distinct flanking anchors')
    from graph_cpu import build
    native=build(reference,observations,k,minimum)
    if native is not None:return native
    return _graph_python(reference,observations,k,minimum)

def _graph_python(reference, observations, k=31, minimum=2):
    by_molecule=collections.defaultdict(set)
    for molecule, sequence in observations:
        sequence=sequence.upper()
        # Enumerate exactly the same canonical windows, without rechecking all
        # k bases for every overlapping word. Ambiguous bases split runs.
        for match in CANONICAL_RUN.finditer(sequence):
            if match.end()-match.start()<k:continue
            words=by_molecule[molecule]
            words.update(sequence[i:i+k] for i in range(match.start(),match.end()-k+1))
    support=collections.Counter(word for words in by_molecule.values() for word in words)
    reference_words={reference[i:i+k] for i in range(len(reference)-k+1)}
    adjacency=collections.defaultdict(list)
    for word in sorted(reference_words | {w for w,n in support.items() if n>=minimum}):
        adjacency[word[:-1]].append((word[-1],support[word],word not in reference_words))
    return adjacency,dict(nodes=len(adjacency),observed_edges=len(support),molecules=len(by_molecule),
        novel_retained_edges=sum(novel for rows in adjacency.values() for base,count,novel in rows))

def assemble(reference, observations, *, k=31, minimum=2, beam=64, max_delta=50):
    reference=reference.upper();adjacency,counts=graph(reference,observations,k,minimum)
    # Edge probabilities depend only on the graph, not on the search path.
    transitions={}
    for node,edges in adjacency.items():
        total=sum(count+1 for base,count,new in edges)
        transitions[node]=[(base,math.log((count+1)/total),new) for base,count,new in edges]
    first=reference[:k-1];last=reference[-(k-1):]
    # Ambiguous anchors cannot define a local locus unambiguously.
    if reference.find(first,1)!=-1 or reference.find(last)!=len(reference)-len(last):
        return [],dict(counts,fallback='nonunique reference anchors')
    lower=max(2*(k-1),len(reference)-max_delta);upper=len(reference)+max_delta
    states=[(first,0.,False)];completed={};expanded=0
    for length in range(k-1,upper+1):
        next_states=[]
        for sequence,score,novel in states:
            if sequence.endswith(last) and length>=lower:
                if novel and sequence!=reference:completed[sequence]=max(score,completed.get(sequence,-math.inf))
                # A unique right reference anchor marks the local path end.
                continue
            if length==upper:continue
            edges=transitions.get(sequence[-(k-1):],[])
            for base,log_probability,new in edges:
                expanded+=1
                next_states.append((sequence+base,score+log_probability,novel or new))
        # Search is bounded independently of truth, phenotype, or sample names.
        states=sorted(next_states,key=lambda item:(-item[1],item[0]))[:beam]
        if not states:break
    ordered=sorted(completed,key=lambda sequence:(-completed[sequence],abs(len(sequence)-len(reference)),sequence))
    return ordered,dict(counts,expanded_states=expanded,completed_paths=len(ordered),k=k,beam=beam,max_delta=max_delta,
        ranking='Graph branch support only; not a calibrated read or genotype likelihood')
