import math
import numpy as np
from learning import pairs,gt_index,probability,proposals,eligible,call_features,kind
from legacy_features import features as original_features
from extra_features import extra
from projection import project
from set_truth_labeler import normalize_biallelic

def candidate_choices(rec,sample,expanded=False):
    selected=set(proposals(rec,sample))
    if not expanded:return np.asarray(sorted(selected),np.int32)
    c=rec.samples[sample];n=len(rec.alleles);order=pairs(n);scores=np.zeros(len(order))
    for key in ['GP','NGP','BGP']:
        p=probability(c,key,len(order))
        if p is not None:scores=np.maximum(scores,p)
    pl=c.get('PL')
    if pl and len(pl)==len(order) and all(x is not None and math.isfinite(x) for x in pl):
        p=np.exp(-np.asarray(pl)*math.log(10)/10);p/=p.sum();scores=np.maximum(scores,p)
    support=c.get('AAD') or c.get('AD')
    if support and len(support)==n and all(v is not None for v in support) and sum(support)>0:
        top=np.argsort(-np.asarray(support),kind='stable')[:4]
        for a in top:
            for b in top:selected.add(gt_index((int(a),int(b))))
    priority=np.argsort(-scores,kind='stable');mass=scores.sum();cumulative=0.
    for j in priority:
        if len(selected)>=24:break
        selected.add(int(j));cumulative+=scores[j]
        if cumulative>=.995*mass and len(selected)>=8:break
    # Original catalogue is unchanged; only genotype hypotheses are expanded.
    return np.asarray(sorted(selected),np.int32)

def allele_events(rec,sequence):
    events=[]
    for alt in rec.alleles:
        mapping,gaps=project(rec.ref,alt);items=set()
        for i,j in enumerate(mapping):
            if j>=0 and rec.ref[i]!=alt[j]:items.add((rec.start+i,rec.ref[i],alt[j]))
        for position,deleted,inserted in gaps:
            p=rec.start+position
            if p<=0:continue
            anchor=sequence[p-1];ref=anchor+deleted;variant=anchor+inserted
            try:key=normalize_biallelic(p-1,ref,variant,sequence)
            except ValueError:key=(p-1,ref,variant)
            items.add(key)
        events.append(items)
    return events

def component_cost(rec,sequence,choices):
    events=allele_events(rec,sequence);keys=sorted(set().union(*events));order=pairs(len(rec.alleles))
    if not keys:return np.zeros((len(choices),len(choices)),np.float32),np.zeros((len(choices),6),np.float32)
    weights=np.asarray([2. if len(r)!=len(a) else 1. for p,r,a in keys])
    dose=np.asarray([[int(k in events[order[j][0]])+int(k in events[order[j][1]]) for k in keys] for j in choices],np.int8)
    # Costs are rows = decision, columns = possible latent truth genotype.
    cost=((dose[:,None,:]!=dose[None,:,:])*weights).sum(2)
    cost/=max(1.,float(weights.sum()))
    is_indel=weights>1;features=[]
    for row in dose:
        features.append([sum((row>0)&~is_indel),sum((row>0)&is_indel),sum(row==2),sum(row==1),sum(row>0),len(keys)])
    return cost.astype(np.float32),np.asarray(features,np.float32)

def feature_rows(rec,sample,sequence,choices):
    c=rec.samples[sample];old=tuple(c['GT']);phase=c.phased;base=call_features(rec,sample,sequence)
    order=pairs(len(rec.alleles));n=len(order);index=gt_index(old);ps=[probability(c,k,n) for k in ['GP','NGP','BGP']]
    rows=[];support=c.get('AAD');usable=support is not None and len(support)==len(rec.alleles) and all(v is not None for v in support) and sum(support)>0
    dp=float(c.get('DP') or 0);adp=float(c.get('ADP') or 0)
    try:
        for j in choices:
            gt=order[j];c['GT']=gt
            ratios=[float(np.clip(math.log(max(1e-12,p[j]))-math.log(max(1e-12,p[index])),-30,30)) if p is not None else 0 for p in ps]
            if usable:
                total=sum(support);a,b=gt;minor=min(support[a],support[b]);major=max(support[a],support[b]);selected=sum(support[k] for k in set(gt))
                aad=[math.log1p(minor),math.log1p(major),minor/total,major/total,selected/total,1-selected/total,float(minor==0),float(minor<2),abs(support[a]/max(1,selected)-.5) if a!=b else 0.]
            else:aad=[-1.]*9
            rows.append(np.r_[base,original_features(rec,sample),float(j==index),float(j==0),ratios,aad,min(2.,adp/max(1,dp)),math.log1p(adp),float(usable),float(len(rec.alleles)>12)])
    finally:c['GT']=old;c.phased=phase
    costs,cf=component_cost(rec,sequence,choices)
    x=np.c_[np.asarray(rows,np.float32),extra(rec,sample,choices),cf].astype(np.float32)
    if not np.isfinite(x).all():raise ValueError('Invalid features')
    return x,costs

def logits(x,weights):
    z=np.clip((x-weights['mean'])/weights['scale'],-8,8)
    for j in range(2):z=np.tanh(z@weights['w'+str(j)].T+weights['b'+str(j)])
    raw=(z@weights['w2'].T+weights['b2']).ravel()
    return np.log(np.maximum(1e-4,x[:,54]))+8*np.tanh(raw/8)

def softmax(z):
    p=np.exp(z-z.max());return p/p.sum()

def decision(q,cost,choices,current,config,reliability,variant_kind):
    original=int(np.flatnonzero(choices==current)[0]);risk=cost@q
    best=int(np.lexsort((-q,risk))[0]) if config['component_decision'] else int(q.argmax())
    gain=float(risk[original]-risk[best]);switch=best!=original and q[best]>=config['min_p'] and gain>=config['min_gain']
    selected=best if switch else original
    # Component consensus can support a GT even when whole-block mass is split.
    confidence=float(1-risk[selected]);threshold={'SNV':.6,'INDEL':.3}[variant_kind]
    abstain=not switch and reliability<threshold
    if config.get('rescue_no_call') and abstain:
        abstain=not (q[selected]>=config['rescue_p'] and confidence>=config['rescue_consensus'])
    return int(choices[selected]),bool(abstain),dict(p=float(q[selected]),component_consensus=confidence,switch=bool(switch),risk_gain=gain)
