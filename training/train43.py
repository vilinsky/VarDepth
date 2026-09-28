"""Fit on chr1-12, calibrate on chr13-14, select thresholds on chr15-16.

Only aggregate diagnostics and a portable JSON model may leave the VM.
Final test labels chr17-22 must remain unopened until MODEL_FROZEN exists.
"""
import collections,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from features43 import NAMES,infer

OUT=Path('/mnt/vdpub14-analysis01/v43-filter')
PARAMS=dict(max_iter=120,learning_rate=.08,max_leaf_nodes=15,max_depth=4,min_samples_leaf=80,
            l2_regularization=5.,max_bins=127,early_stopping=False,random_state=43017)
CASES=['lowpass','wes','hybrid','wgs']

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def export_model(m):
    trees=[]
    for predictor in m._predictors:
        assert len(predictor)==1
        tree=[]
        for n in predictor[0].nodes:
            assert not n['is_categorical']
            threshold=0. if n['is_leaf'] else float(np.clip(n['num_threshold'],-1e300,1e300))
            tree.append(dict(leaf=bool(n['is_leaf']),value=float(n['value']),feature=int(n['feature_idx']),
                             threshold=threshold,missing_left=bool(n['missing_go_to_left']),left=int(n['left']),right=int(n['right'])))
        trees.append(tree)
    return dict(baseline=float(m._baseline_prediction[0,0]),trees=trees)

def load(bounds):
    values=collections.defaultdict(list);sources=[]
    for case in CASES:
        for n in bounds:
            folder=OUT/'features'/case/('chr'+str(n))
            if not (folder/'COMPLETE.json').exists():continue
            receipt=json.loads((folder/'COMPLETE.json').read_text());assert receipt['success']
            with np.load(folder/'data.npz') as d:
                for k in d.files:values[k].append(d[k])
                values['case'].append(np.full(len(d['y']),CASES.index(case),np.int8))
            sources.append(dict(case=case,chrom='chr'+str(n),receipt_sha256=digest(folder/'COMPLETE.json')))
    return {k:np.concatenate(v) for k,v in values.items()},sources

def f1(y,accept):
    tp=int(np.sum(y&accept));fp=int(np.sum(~y&accept));fn=int(np.sum(y&~accept))
    return 2*tp/max(1,2*tp+fp+fn)

def choose_threshold(y,p,cases,mask):
    # Equal weight for every available assay within the depth group. This is
    # conditional classifier F1, not genomic hap.py F1 (existing FNs excluded).
    indices=[mask&(cases==i) for i in range(len(CASES))]
    indices=[m for m in indices if m.sum()>=100 and y[m].sum()>=50]
    if not indices:return None
    candidates=[0.,*(.05+np.arange(95)*.01).tolist(),.999]  # fixed grid before test access
    rows=[]
    for threshold in candidates:
        rows.append((float(np.mean([f1(y[m],p[m]>=threshold) for m in indices])),float(threshold)))
    best=max(rows,key=lambda z:(z[0],-z[1]))
    return dict(threshold=best[1],conditional_macro_f1=best[0],n=int(mask.sum()),
                tp_labels=int(y[mask].sum()),fp_labels=int((~y[mask]).sum()))

def main():
    start=time.monotonic();cpu=time.process_time()
    assert (OUT/'EXTRACTION_DEVELOPMENT.json').exists()
    assert not (OUT/'MODEL_FROZEN.json').exists()
    train,train_sources=load(range(1,13));cal,cal_sources=load(range(13,15));tune,tune_sources=load(range(15,17))
    model=dict(version='0.43.0-dev1',feature_names=NAMES,parameters=PARAMS,
               train_chromosomes=list(range(1,13)),probability_calibration_chromosomes=[13,14],
               threshold_chromosomes=[15,16],test_chromosomes=list(range(17,23)),
               coverage_cut=10.,coverage_measure='max(FORMAT/DP,FORMAT/ADP); missing falls back to low coverage',
               context_window_bp=50000,context_minimum_anchors=20,classes={},experimental=True,
               genotype_evidence_unchanged=True,truth_free_application=True,development_sample='HG005',independent_person_validation=False)
    reports=[]
    with threadpool_limits(limits=8):
      for kind,name in [(0,'SNV'),(1,'INDEL')]:
        a=train['kind']==kind;b=cal['kind']==kind;t=tune['kind']==kind
        x=train['x'][a];y=train['y'][a];case=train['case'][a]
        assert len(y)>2000 and np.sum(y==0)>100 and np.sum(y==1)>100
        weight=np.ones(len(y))
        for i in np.unique(case):weight[case==i]=len(y)/(len(np.unique(case))*np.sum(case==i))
        clf=HistGradientBoostingClassifier(**PARAMS).fit(x,y,sample_weight=weight)
        exported=export_model(clf)
        cx=cal['x'][b];cy=cal['y'][b];tx=tune['x'][t];ty=tune['y'][t].astype(bool)
        cp=clf.predict_proba(cx)[:,1];tp=clf.predict_proba(tx)[:,1]
        # Check portable scorer across each assay, including rows with NaNs.
        sample=np.linspace(0,len(tx)-1,min(20000,len(tx))).astype(int)
        error=float(np.max(np.abs(infer(exported,tx[sample])-tp[sample])))
        assert error<1e-6,error
        calibrator=IsotonicRegression(out_of_bounds='clip').fit(cp,cy)
        probability=calibrator.predict(tp)
        exported['calibration']=dict(x=calibrator.X_thresholds_.tolist(),y=calibrator.y_thresholds_.tolist())
        effective=np.maximum(np.nan_to_num(np.expm1(tx[:,0]),nan=0),np.nan_to_num(np.expm1(tx[:,1]),nan=0))
        thresholds={}
        global_choice=choose_threshold(ty,probability,tune['case'][t],np.ones(len(ty),bool))
        for depth_name,mask in [('low',effective<10),('high',effective>=10)]:
            choice=choose_threshold(ty,probability,tune['case'][t],mask)
            if choice is None or choice['n']<500 or choice['fp_labels']<30:
                choice=dict(global_choice,fallback='pooled threshold; insufficient calibration labels in depth group')
            thresholds[depth_name]=choice
        exported['thresholds']={k:v['threshold'] for k,v in thresholds.items()}
        model['classes'][name]=exported
        report=dict(kind=name,train_rows=len(y),calibration_rows=len(cy),threshold_rows=len(ty),
                    portable_probability_max_error=error,thresholds=thresholds,
                    conditional_only=True,no_test_metrics_seen=True)
        reports.append(report);print(json.dumps(report),flush=True)
    raw=(json.dumps(model,separators=(',',':'),allow_nan=False)+'\n').encode()
    (OUT/'model43.json').write_bytes(raw)
    receipt=dict(success=True,model_sha256=hashlib.sha256(raw).hexdigest(),reports=reports,
                 training_sources=train_sources,calibration_sources=cal_sources,threshold_sources=tune_sources,
                 features_source_sha256=digest(Path(__file__).parent/'features43.py'),train_source_sha256=digest(__file__),
                 wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
                 test_labels_accessed=False,params=PARAMS,threshold_objective='Conditional query classification macro-F1; full genomic metrics require fresh hap.py')
    (OUT/'MODEL_FROZEN.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
