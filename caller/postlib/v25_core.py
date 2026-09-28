import collections,math
import numpy as np
from core import allele_events as legacy_allele_events,candidate_choices,feature_rows,pairs,gt_index,eligible
from learning import call_features,probability,max_run
from extra_features import allele_properties
from projection import project
from set_truth_labeler import normalize_biallelic

def allele_events(rec,sequence):
    # Identical sequence contexts must not change primitive-event boundaries.
    output=[]
    for alt in rec.alleles:
        ref=rec.ref;start=rec.start;left=0
        while left<min(len(ref),len(alt)) and ref[left]==alt[left]:left+=1
        right=0
        while right<min(len(ref)-left,len(alt)-left) and ref[-1-right]==alt[-1-right]:right+=1
        rr=ref[left:len(ref)-right if right else len(ref)];aa=alt[left:len(alt)-right if right else len(alt)];p=start+left;events=set()
        if not rr and not aa:output.append(events);continue
        if not rr or not aa:
            if p<=0:raise ValueError('Unanchored chromosome-start event')
            anchor=sequence[p-1];events.add(normalize_biallelic(p-1,anchor+rr,anchor+aa,sequence))
        else:
            mapping,gaps=project(rr,aa)
            for i,j in enumerate(mapping):
                if j>=0 and rr[i]!=aa[j]:events.add((p+i,rr[i],aa[j]))
            for offset,deleted,inserted in gaps:
                x=p+offset
                if x<=0:raise ValueError('Unanchored chromosome-start event')
                anchor=sequence[x-1];events.add(normalize_biallelic(x-1,anchor+deleted,anchor+inserted,sequence))
        output.append(events)
    return output

def short_event(k):
    p,r,a=k
    return (len(r)==len(a)==1 and r!=a) or (len(r)!=len(a) and max(len(r)-1,len(a)-1)<=50)

def event_data(rec,sample,sequence):
    events=allele_events(rec,sequence);keys=sorted(k for k in set().union(*events) if short_event(k))
    if not keys:return None
    order=pairs(len(rec.alleles));dose=np.asarray([[int(k in events[a])+int(k in events[b]) for k in keys] for a,b in order],np.int8)
    base=call_features(rec,sample,sequence);c=rec.samples[sample];rows=[];ng=len(order)
    distributions=[probability(c,key,ng) for key in ['GP','NGP','BGP']]
    pl=c.get('PL')
    if pl and len(pl)==ng and all(v is not None and math.isfinite(v) for v in pl):
        x=np.asarray(pl,float);x=np.exp(-(x-x.min())*math.log(10)/10);distributions.append(x/x.sum())
    else:distributions.append(None)
    original=gt_index(c['GT'])
    for j,k in enumerate(keys):
        p,r,a=k;carrier=np.asarray([k in e for e in events]);row=list(base)+list(allele_properties(r,a))
        row.extend([float(len(r)!=len(a)),float(len(a)>len(r)),math.log1p(abs(len(a)-len(r))),float(dose[original,j]),float(carrier.sum()),float(len(keys))])
        for q in distributions:row.extend(np.bincount(dose[:,j],weights=q,minlength=3) if q is not None else [-1.]*3)
        for field in ['AAD','AD']:
            v=c.get(field)
            if v and len(v)==len(carrier) and all(x is not None and x>=0 for x in v) and sum(v)>0:
                z=np.asarray(v,float);alt=z[carrier].sum();ref=z[~carrier].sum();row.extend([math.log1p(alt),math.log1p(ref),alt/(alt+ref)])
            else:row.extend([-1.]*3)
        context=sequence[max(0,p-16):p+len(r)+16];row.extend([max_run(context,k) for k in [1,2,3,4]])
        rows.append(row)
    x=np.asarray(rows,np.float32);assert np.isfinite(x).all()
    return keys,dose,x

def choose(dose,q,choices,current,min_confidence,min_gain,indel,indel_weight=1.):
    """Expected component error, constrained to a whole retained diploid GT.

    No independence/product assumption, no wholesale block rejection, and no
    rescue of baseline no-calls. Gain is averaged over changed components.
    """
    if current is None:return None,dict(reason='retained_no_call')
    weights=np.where(indel,indel_weight,1.);loss=1-q[np.arange(len(q))[None,:],dose[choices]]
    risk=loss@weights;oldpos=int(np.flatnonzero(choices==current)[0]);best=int(np.argmin(risk));selected=int(choices[best]);changed=dose[selected]!=dose[current]
    if not changed.any():return current,dict(reason='same_components')
    confidence=float(q[np.flatnonzero(changed),dose[selected,changed]].min());gain=float((risk[oldpos]-risk[best])/weights[changed].sum())
    accepted=confidence>=min_confidence and gain>=min_gain
    return selected if accepted else current,dict(reason='changed' if accepted else 'retained_uncertain',gain=gain,min_component_confidence=confidence,changed_components=int(changed.sum()))
