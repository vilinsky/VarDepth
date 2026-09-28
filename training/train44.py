"""Coverage calibration and conservative genomic-F1 candidate screening.

Only chr13-16 are accessed. Existing chr1-12 trees are reused unchanged.
The approximate screening objective MUST be followed by real hap.py selection.
"""
import copy, hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
from features43 import infer
from policy44 import calibrated, depth_groups, genomic_f1

PARENT = Path('/mnt/vdpub14-analysis01/v43-filter')
BASE = Path('/mnt/vdpub14-analysis01/v41-release-benchmark')
OUT = Path('/mnt/vdpub14-analysis01/v44-filter02')
CASES = ['lowpass', 'wes', 'hybrid', 'wgs']
CONTEXTS = [('all', 0), ('capture', 1), ('outside', 4)]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(chroms):
    values = {}; sources = []
    for case_id, case in enumerate(CASES):
        for n in chroms:
            folder = PARENT/'features'/case/f'chr{n}'
            if not (folder/'COMPLETE.json').exists(): continue
            receipt = json.loads((folder/'COMPLETE.json').read_text())
            assert receipt['success'] and receipt['training_sampling'] is None
            with np.load(folder/'data.npz') as data:
                for k in data.files: values.setdefault(k, []).append(data[k])
                values.setdefault('case', []).append(np.full(len(data['y']), case_id, np.int8))
            sources.append(dict(case=case, chrom=f'chr{n}', sha256=sha(folder/'data.npz')))
    return {k: np.concatenate(v) for k, v in values.items()}, sources


def baselines(kind, sources):
    result = []
    for case_id, case in enumerate(CASES):
        for context, bit in CONTEXTS:
            counts = np.zeros(4, float)
            for source in sources:
                if source['case'] != case: continue
                rows = json.loads((BASE/'assessment/vardepth'/case/source['chrom']/'ALL/COUNTS.json').read_text())['rows']
                row = next(r for r in rows if r['kind'] == kind and r['context'] == context)
                counts += [row[k] for k in ['query_TP','truth_TP','FP','FN']]
            if counts[1]+counts[3] > 0:
                result.append((case_id, context, bit, counts))
    return result


def optimize(y, probabilities, groups, cases, regions, scope, grid, risk):
    """Preserve baseline FN and asymmetric TP; count only matched removed rows.

    Approximate removed truth TP via within-scope truth/query TP ratio.
    Unmatched/unsupported records cannot be silently assumed correctly labelled.
    Their effect is checked in fresh hap.py, never reported as screening accuracy.
    """
    baseline = np.asarray([s[3] for s in scope])
    baseline_f1 = genomic_f1(*baseline.T)
    removed = np.zeros((6, len(grid), len(scope), 2))
    for j, (case, _, bit, _) in enumerate(scope):
        mask = cases == case
        if bit: mask &= (regions & bit) != 0
        for g in range(6):
            for k, cut in enumerate(grid):
                rejected = mask & (groups == g) & (probabilities < cut)
                removed[g,k,j] = [np.sum(rejected & y), np.sum(rejected & ~y)]
    ratios = np.divide(baseline[:,1], baseline[:,0], out=np.ones(len(scope)), where=baseline[:,0]>0)

    def assess(index):
        loss = removed[np.arange(6), index].sum(axis=0)
        qloss = np.minimum(baseline[:,0], loss[:,0] * risk)
        tloss = np.minimum(baseline[:,1], loss[:,0] * ratios * risk)
        fp_removed = np.minimum(baseline[:,2], loss[:,1])
        score = genomic_f1(baseline[:,0]-qloss, baseline[:,1]-tloss,
                           baseline[:,2]-fp_removed, baseline[:,3]+tloss)
        # Strict per-scope safety constraint on calibration, with explicit zero
        # threshold fallback. No test or comparator results enter this search.
        feasible = bool(np.all(score >= baseline_f1-1e-12))
        return float(np.mean(score-baseline_f1)) if feasible else -1., score

    index = np.zeros(6, int)
    for _ in range(3):
        changed = False
        for g in range(6):
            best = (assess(index)[0], -grid[index[g]], int(index[g]))
            for k in range(len(grid)):
                proposal = index.copy(); proposal[g] = k
                candidate = (assess(proposal)[0], -grid[k], k)
                if candidate > best: best = candidate
            changed |= best[2] != index[g]; index[g] = best[2]
        if not changed: break
    objective, f1 = assess(index)
    return [grid[i] for i in index], dict(surrogate_macro_gain=objective,
        scopes=[dict(case=CASES[s[0]],context=s[1],baseline_f1=float(b),surrogate_f1=float(f))
                for s,b,f in zip(scope,baseline_f1,f1)],risk_multiplier=risk)


def main():
    start=time.monotonic();cpu=time.process_time()
    registration=json.loads((OUT/'REGISTRATION.json').read_text())
    assert not (OUT/'CANDIDATES_READY.json').exists()
    parent=json.loads((PARENT/'model43.json').read_text())
    assert sha(PARENT/'model43.json')=='4f8aee970249e693baaa47f9ecf28714ca73a4dfb9c5c9b2a6517cd8034a7c87'
    cal,cal_sources=load([13,14]);tune,tune_sources=load([15,16])
    model=copy.deepcopy(parent);model.update(version='0.44.0-dev1',coverage_bins=registration['coverage_bins'],
        regression_previously_seen=True,independent_person_validation=False,
        model_training='v43 frozen trees reused; v44 depth calibration and genomic-F1 thresholds',
        zero_threshold_probability='not computed; V44P missing, not probability one',
        unmodelled_filter_policy='inherit v42 sensitivity: promote LowQual, preserve hard filters and genotype evidence')
    reports=[];candidates={name:copy.deepcopy(model) for name in registration['candidates']}
    for kind,name,happy_kind in [(0,'SNV','SNP'),(1,'INDEL','INDEL_1_50')]:
        mask=cal['kind']==kind;cx=cal['x'][mask];cy=cal['y'][mask]
        ccase=cal['case'][mask];cgroups=depth_groups(cx,model['coverage_bins']);raw=infer(parent['classes'][name],cx)
        m=model['classes'][name];m['depth_calibrations']={};cal_reports=[]
        for group in range(6):
            selected=cgroups==group;pos=int(cy[selected].sum());neg=int(selected.sum())-pos
            fallback=selected.sum()<1000 or neg<30 or pos<100
            if fallback: calibration=copy.deepcopy(m['calibration'])
            else:
                weights=np.ones(int(selected.sum()));ids=ccase[selected]
                for c in np.unique(ids):weights[ids==c]=len(ids)/(len(np.unique(ids))*np.sum(ids==c))
                iso=IsotonicRegression(out_of_bounds='clip').fit(raw[selected],cy[selected],sample_weight=weights)
                calibration=dict(x=iso.X_thresholds_.tolist(),y=iso.y_thresholds_.tolist())
            m['depth_calibrations'][str(group)]=calibration
            cal_reports.append(dict(group=group,rows=int(selected.sum()),TP=pos,FP=neg,pooled_fallback=bool(fallback)))
        mask=tune['kind']==kind;x=tune['x'][mask];y=tune['y'][mask].astype(bool)
        groups=depth_groups(x,model['coverage_bins']);probability=calibrated(m,x,groups)
        for candidate,risk in [('genomic_f1',1.),('recall_guard',3.),('baseline',1.)]:
            thresholds,report=optimize(y,probability,groups,tune['case'][mask],tune['region'][mask],
                baselines(happy_kind,tune_sources),registration['threshold_grid'],risk)
            if candidate=='baseline':thresholds=[0.]*6;report=dict(explicit_unfiltered_baseline=True)
            entry=copy.deepcopy(m);entry['depth_thresholds']=thresholds
            candidates[candidate]['classes'][name]=entry
            reports.append(dict(candidate=candidate,kind=name,thresholds=thresholds,calibration=cal_reports,**report))
    receipt=dict(success=True,reports=reports,calibration_sources=cal_sources,threshold_sources=tune_sources,
        wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,regression_labels_accessed=False,candidates=[])
    for name,m in candidates.items():
        dest=OUT/'candidates'/name;dest.mkdir(parents=True,exist_ok=False)
        (dest/'model44.json').write_text(json.dumps(m,separators=(',',':'),allow_nan=False)+'\n')
        receipt['candidates'].append(dict(name=name,sha256=sha(dest/'model44.json')))
    (OUT/'CANDIDATES_READY.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(success=True,stage='calibrated_candidates_ready',reports=reports)),flush=True)

if __name__=='__main__':main()
