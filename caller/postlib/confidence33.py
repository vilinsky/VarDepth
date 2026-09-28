import collections,hashlib,json,math,pickle
from pathlib import Path
import numpy as np,pysam
from threadpoolctl import threadpool_limits
import pipeline32 as p

def allele_digest(rec):
    return hashlib.sha256(json.dumps([rec.chrom,rec.start,rec.alleles]).encode()).hexdigest()

def numeric(value,default=-1.):
    if value is None:return default
    try:return float(value) if math.isfinite(float(value)) else default
    except (ValueError,TypeError):return default

def features(sample,arm,with_labels=False):
    out=ROOT/'confidence_features_v2'/arm/sample/'features.npz'
    if out.exists():d=load(out)
    else:
        source=calls(sample,arm);base=data(sample);q=student_probabilities(sample,arm,base)
        index={s:i for i,s in enumerate(base['ids'])};rows=[];ids=[];digests=[];pos=[];kind=[];needs=[];gtids=[];nonref=[];oldpass=[]
        with pysam.VariantFile(source) as f:
            for rec in f:
                i=index.get(rec.id);c=rec.samples[sample];gt=c.get('GT')
                if i is None or not gt or None in gt:continue
                a,b=base['offsets'][i:i+2];da,db=base['dose_offsets'][i:i+2]
                dose=base['dose'][da:db].reshape(-1,b-a);gi=p.gt_index(gt)
                if gi>=len(dose):raise ValueError('Genotype outside frozen allele catalogue')
                qr=q[a:b];sel=qr[np.arange(b-a),dose[gi]];risk=(1-qr[np.arange(b-a)[None,:],dose]).sum(1)
                ordered=np.sort(risk);gap=ordered[1]-ordered[0] if len(ordered)>1 else 0.
                oldgq=numeric(c.get('GQ'));oldqual=numeric(rec.qual)
                summary=[math.log1p(b-a),math.log1p(len(rec.alleles)),float(sel.min()),float(sel.mean()),
                         float(sel.max()),float(sel.std()),float(risk[gi]/max(1,b-a)),float(gap/max(1,b-a)),
                         float(qr[:,3].max()),float(np.mean(base['indel'][a:b])),math.log1p(max(0,numeric(c.get('DP'),0))),
                         oldgq,oldqual,float(any(gt)),float(gt[0]!=gt[1]),
                         math.log1p(len(rec.ref)),math.log1p(max(map(len,rec.alleles))),
                         float(c.get('R30OGT') is not None),float(c.get('R24OGT') is not None),
                         numeric(c.get('R30MC')),numeric(c.get('R30GAIN'))]
                rows.append(np.r_[base['x'][a:b].mean(0),qr.mean(0),summary]);ids.append(rec.id);digests.append(allele_digest(rec));pos.append(rec.start)
                kind.append(bool(base['indel'][a:b].any()));gtids.append(gi);nonref.append(any(gt))
                filt=set(rec.filter);oldpass.append(not filt or filt=={'PASS'})
                # Earlier R21 correction emits conditional confidence even when
                # present. Recalibrate all LowQual records touched by correction,
                # not just the later corrections which explicitly cleared scores.
                corrected=any(c.get(k) is not None for k in ['R21OGT','R24OGT','R30OGT'])
                needs.append(filt=={'LowQual'} and (corrected or oldqual<0 or oldgq<0))
        d=dict(x=np.asarray(rows,np.float32),ids=np.asarray(ids),digests=np.asarray(digests),pos=np.asarray(pos),
               indel=np.asarray(kind),needs=np.asarray(needs),gt=np.asarray(gtids),nonref=np.asarray(nonref),oldpass=np.asarray(oldpass))
        assert np.isfinite(d['x']).all();out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**d)
        previous=ROOT/'confidence_features'/arm/sample/'features.npz'
        if previous.exists():
            old=load(previous)
            assert np.array_equal(old['x'],d['x']) and np.array_equal(old['ids'],d['ids']), 'Scope revision changed model features'
        save(out.parent/'COMPLETE.json',dict(rows=len(ids),features=d['x'].shape[1],eligible_corrections=int(d['needs'].sum()),
             source_sha256=sha(source),truth_used=False))
    if with_labels:
        labels=truth_labels(sample);y=np.full((len(d['ids']),2),-1,np.int8)
        for i,rid in enumerate(d['ids']):
            if rid in labels:
                gs,var,digest=labels[rid];assert digest==d['digests'][i]
                y[i]=[int(d['gt'][i] in gs),var]
        d['y']=y
    return d

def sigmoid(v):return 1/(1+np.exp(-np.clip(v,-40,40)))

def logit(v):v=np.clip(v,1e-6,1-1e-6);return np.log(v/(1-v))

def predict(sample,arm,d=None):
    d=d if d is not None else features(sample,arm)
    with (MODELS/'confidence33.pkl').open('rb') as f:models=pickle.load(f)
    probs=np.zeros((len(d['x']),2))
    with threadpool_limits(1):
        for kind in [False,True]:
            mask=d['indel']==kind
            if not mask.any():continue
            for endpoint in [0,1]:
                m,slope,intercept=models[kind,endpoint];raw=m.predict_proba(d['x'][mask])[:,1]
                probs[mask,endpoint]=sigmoid(slope*logit(raw)+intercept)
    # Coherence: confidence of a nonref GT cannot exceed P(nonref).
    # The inference module only updates missing-confidence nonref records.
    probs[d['nonref'],1]=np.maximum(probs[d['nonref'],1],probs[d['nonref'],0])
    probs=np.clip(probs,1e-6,1-1e-6)
    return probs

def write(sample,arm):
    out=ROOT/'calibrated-v2'/arm/sample/'calls.vcf.gz'
    if out.exists() and Path(str(out)+'.json').exists():return out
    source=calls(sample,arm);d=features(sample,arm);prob=predict(sample,arm,d);threshold={'SNV':0.5,'INDEL':0.5}
    index={rid:i for i,rid in enumerate(d['ids'])};stats=collections.Counter();before=hashlib.sha256();after=hashlib.sha256()
    out.parent.mkdir(parents=True,exist_ok=True)
    with pysam.VariantFile(source) as f:
        h=f.header.copy()
        for name,desc in [('V33PC','Calibrated probability that the final diploid GT is correct; binary endpoint, not joint GP'),
                          ('V33PV','Calibrated probability of a nonreference genotype')]:
            if name not in h.formats:h.formats.add(name,1,'Float',desc)
        h.add_meta('VarDepth_confidence_model',value='v33-'+arm+'-Illumina30x-research')
        with pysam.VariantFile(out,'wz',header=h) as dst:
            for rec in f:
                rec.translate(h);c=rec.samples[sample];gt=c.get('GT');i=index.get(rec.id)
                identity=json.dumps([rec.chrom,rec.start,rec.alleles,gt,c.phased,c.get('PS')]).encode();before.update(identity)
                stats['records']+=1
                if i is not None and d['needs'][i] and d['nonref'][i]:
                    pc,pv=map(float,prob[i]);c['V33PC']=pc;c['V33PV']=pv
                    c['GQ']=int(round(-10*math.log10(1-pc)));rec.qual=-10*math.log10(1-pv)
                    # Binary correctness/presence probabilities do not identify
                    # a joint distribution over the catalogue: truth may carry
                    # an allele outside it. Never retain stale conditional GP/DS.
                    for field in ['GP','DS']:
                        if field in c:
                            n=len(rec.alleles)*(len(rec.alleles)+1)//2 if field=='GP' else len(rec.alleles)-1
                            c[field]=tuple([None]*n)
                    t=threshold['INDEL' if d['indel'][i] else 'SNV'];rec.filter.clear();rec.filter.add('PASS' if min(pc,pv)>=t else 'LowQual')
                    stats['confidence_updated']+=1;stats['new_PASS' if min(pc,pv)>=t else 'remains_LowQual']+=1
                after.update(json.dumps([rec.chrom,rec.start,rec.alleles,c.get('GT'),c.phased,c.get('PS')]).encode());dst.write(rec)
    assert before.digest()==after.digest(),'Calibration changed genotype, allele or phase'
    pysam.tabix_index(str(out),preset='vcf',force=True)
    save(str(out)+'.json',dict(sample=sample,arm=arm,stats=dict(stats),thresholds=threshold,GT_alleles_phase_identical=True,
         identity_sha256=before.hexdigest(),source_sha256=sha(source),output_sha256=sha(out),truth_access=False,
         joint_GP_and_dosage_not_fabricated=True,stale_GP_DS_cleared=True,scope_revision=2))
    return out
