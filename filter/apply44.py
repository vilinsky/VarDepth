"""Apply learned coverage-aware acceptance without changing genotype evidence."""
import argparse,collections,hashlib,json,math,time
from pathlib import Path
import numpy as np
import pysam
from features43 import NAMES,depth_profile,key_and_kind,features
from policy44 import decisions

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def invariant(rec):
    info={k:rec.info[k] for k in rec.info if k!='V44OF'}
    calls={s:{k:v for k,v in c.items() if k!='V44P'} for s,c in rec.samples.items()}
    phase={s:c.phased for s,c in rec.samples.items()}
    return json.dumps([rec.chrom,rec.start,rec.id,rec.alleles,rec.qual,info,calls,phase],sort_keys=True,separators=(',',':')).encode()+b'\n'
def apply(source,target,model_path):
    start=time.monotonic();cpu=time.process_time();source=Path(source);target=Path(target)
    if target.exists():raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    model=json.loads(Path(model_path).read_text());assert model['feature_names']==NAMES
    profile=depth_profile(source,model['context_window_bp']);before=hashlib.sha256();after=hashlib.sha256()
    counts=collections.Counter();pending=[]
    with pysam.VariantFile(source) as f:
        if len(f.header.samples)!=1:raise ValueError('Exactly one sample required')
        sample=list(f.header.samples)[0];h=f.header.copy()
        if 'V44OF' in h.info or 'V44P' in h.formats:raise ValueError('Input already processed with v44')
        h.info.add('V44OF',1,'String','Original filter before learned coverage-aware acceptance')
        h.formats.add('V44P',1,'Float','Experimental depth-calibrated genotype correctness score; missing if unconditional acceptance skipped scoring; not GP or clinical confidence')
        h.filters.add('V44LowConfidence',None,None,'Below learned acceptance threshold for variant type and measured coverage')
        h.add_meta('VarDepth_filter_model_sha256',value=sha(model_path))
        h.add_meta('VarDepth_filter_development_sample',value='HG005;independent_validation_not_established')
        with pysam.VariantFile(target,'wz',header=h) as writer:
            def flush():
                probabilities={}
                for kind,name in [(0,'SNV'),(1,'INDEL')]:
                    indices=[i for i,(_,k,_,_) in enumerate(pending) if k==kind]
                    if not indices:continue
                    x=np.asarray([pending[i][2] for i in indices],np.float32)
                    m=model['classes'][name]
                    accepted,p,groups=decisions(m,x,model['coverage_bins'])
                    counts['scored']+=int(np.isfinite(p).sum())
                    counts['unconditional_acceptance_score_skipped']+=int(np.isnan(p).sum())
                    for i,prob,passed in zip(indices,p,accepted):probabilities[i]=(float(prob),bool(passed))
                for i,(r,kind,x,original) in enumerate(pending):
                    before.update(invariant(r));r.translate(h)
                    if i in probabilities:
                        probability,passed=probabilities[i];r.info['V44OF']=original or '.'
                        r.samples[sample]['V44P']=probability if math.isfinite(probability) else None
                        r.filter.clear();r.filter.add('PASS' if passed else 'V44LowConfidence')
                        counts[('SNV' if kind==0 else 'INDEL')+('_accepted' if passed else '_rejected')]+=1
                        counts['promoted']+=passed and original not in ['PASS','.','']
                        counts['demoted']+=not passed and original in ['PASS','.','']
                    else:
                        if original=='LowQual':
                            # An unmodelled record inherits the v42 sensitivity
                            # baseline; absence of a model score is not rejection.
                            r.info['V44OF']=original;r.filter.clear();r.filter.add('PASS')
                            counts['unmodelled_inherit_sensitivity']+=1
                        else:counts['unchanged_unsupported_or_hard_filter']+=1
                    after.update(invariant(r));writer.write(r);counts['records']+=1
                pending.clear()
            for rec in f:
                original=(rec.info.get('V42OF') if 'V42OF' in f.header.info else None) or ';'.join(rec.filter)
                if isinstance(original,tuple):original=';'.join(original)
                _,kind=key_and_kind(rec,sample)
                if set(original.split(';'))-{'PASS','.','','LowQual'}:kind=None
                x=features(rec,sample,profile.get((rec.chrom,rec.start//50000),float('nan'))) if kind is not None else None
                pending.append((rec,kind,x,original))
                if len(pending)>=4096:flush()
            flush()
    assert before.digest()==after.digest(),'Genotype, original quality or phase changed'
    pysam.tabix_index(str(target),preset='vcf',force=False)
    receipt=dict(success=True,version='0.44.0-dev1',counts=dict(counts),source_sha256=sha(source),output_sha256=sha(target),
                 model_sha256=sha(model_path),invariant_sha256=before.hexdigest(),genotype_quality_phase_preserved=True,
                 wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,truth_used=False)
    Path(str(target)+'.filter.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','output','model']:p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(apply(a.source,a.output,a.model)))
if __name__=='__main__':main()
