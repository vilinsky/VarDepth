"""Truth-free repair policies: native abstention rescue and coverage-domain guard.

The guard selects a complete source record. It does not manufacture probabilities,
combine incompatible genotype posteriors, or select calls using benchmark labels.
"""
import collections,hashlib,itertools,json,math,sys,time
from pathlib import Path
import pysam

POLICIES=('abstention-rescue','coverage-domain-guard')

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def called(gt):return bool(gt) and all(a is not None for a in gt)

def domain_mode(window,guard):
    depth=window['median_DP']
    if window['anchors']<guard['minimum_anchors'] or depth is None:return 'native','sparse'
    if not math.isfinite(depth) or depth<0:raise ValueError('Invalid coverage profile')
    if depth<guard['lower']:return 'native','below_training_domain'
    if depth>guard['upper']:return 'native','above_training_domain'
    return 'current','within_training_domain'

def select_source(policy,native,current,sample,window,guard):
    if policy=='coverage-domain-guard':return domain_mode(window,guard)
    if policy!='abstention-rescue':raise ValueError('Unknown repair policy')
    n=native.samples[sample];c=current.samples[sample]
    if called(n.get('GT')) and set(native.filter)=={'PASS'} and not called(c.get('GT')) and c.get('R21OGT') is not None:
        return 'native','native_PASS_after_correction_abstention'
    return 'current','unchanged'

def catalogue(rec):return rec.chrom,rec.start,rec.alleles

def decision(rec,sample):
    c=rec.samples[sample]
    return json.dumps([rec.chrom,rec.start,rec.stop,rec.alleles,c.get('GT'),c.phased,c.get('PS'),rec.qual,sorted(rec.filter)],separators=(',',':')).encode()+b'\n'

def retained_format(rec,sample):
    c=rec.samples[sample]
    # Confidence and phase come from the very same source as the selected GT.
    return {k:c.get(k) for k in ['GT','GQ','DP','GP','DS','PS','V33PC','V33PV'] if k in c},c.phased

def repair(case,profile,guard,outroot,policies=POLICIES):
    begin=time.monotonic();sample=case['sample'];native_path=Path(case['native']);current_path=Path(case['current']);outroot=Path(outroot)
    assert profile['source_sha256']==sha(native_path),'Coverage profile is for a different native VCF'
    source_hashes={'native':profile['source_sha256'],'current':sha(current_path)}
    assert set(policies)<=set(POLICIES) and policies
    outroot.mkdir(parents=True,exist_ok=True);marker=outroot/'COMPLETE.json'
    if marker.exists():
        previous=json.loads(marker.read_text());assert previous['source_sha256']==source_hashes and previous['guard']==guard
        assert previous['code_sha256']==sha(__file__),'Completed repair used a different implementation'
        for name,row in previous['policies'].items():assert sha(row['output'])==row['sha256']
        return previous
    width=guard['window_bp'];windows={w['index']:w for w in profile['windows']}
    source_digests={k:hashlib.sha256() for k in ['native','current']}
    output_digests={p:hashlib.sha256() for p in policies};stats={p:collections.Counter() for p in policies}
    output_paths={p:outroot/p/'calls.vcf.gz' for p in policies}
    streams={};count=0
    with pysam.VariantFile(native_path) as original,pysam.VariantFile(current_path) as corrected:
        assert list(original.header.samples)==list(corrected.header.samples)==[sample]
        header=corrected.header.copy()
        for name,description in [('V36SRC','Complete record source selected without truth: native or current'),
                                 ('V36REASON','Registered model-domain guard or abstention recovery decision')]:
            assert name not in header.info;header.info.add(name,1,'String',description)
        header.add_meta('VarDepth_v36',value='research domain guard; original source GT and confidence retained; no new joint GP estimated')
        try:
            for policy,path in output_paths.items():
                path.parent.mkdir(parents=True,exist_ok=True)
                streams[policy]=pysam.VariantFile(str(path)+'.partial','wz',header=header)
            for n,c in itertools.zip_longest(original,corrected):
                assert n is not None and c is not None,'Source VCF lengths differ; no silent catalogue merge'
                assert catalogue(n)==catalogue(c),'Source catalogues differ; repair requires identical ordered alleles'
                assert n.chrom==case['chrom'],'Unexpected chromosome outside the registered repair scope'
                count+=1;nd=decision(n,sample);cd=decision(c,sample)
                source_digests['native'].update(nd);source_digests['current'].update(cd)
                window=windows[n.start//width]
                for policy,dst in streams.items():
                    selected,reason=select_source(policy,n,c,sample,window,guard)
                    source=n if selected=='native' else c;expected=nd if selected=='native' else cd
                    output=source.copy();preserved=retained_format(source,sample);output.translate(header)
                    output.info['V36SRC']=selected;output.info['V36REASON']=reason
                    assert decision(output,sample)==expected and retained_format(output,sample)==preserved
                    dst.write(output);output_digests[policy].update(expected)
                    stats[policy][selected]+=1;stats[policy][reason]+=1
                    stats[policy]['changed_vs_current']+=int(expected!=cd)
        finally:
            for f in streams.values():f.close()
    source_digests={k:v.hexdigest() for k,v in source_digests.items()};results={}
    for policy,path in output_paths.items():
        Path(str(path)+'.partial').replace(path);pysam.tabix_index(str(path),preset='vcf',force=True)
        digest=output_digests[policy].hexdigest()
        results[policy]=dict(output=str(path),sha256=sha(path),decision_sha256=digest,stats=dict(stats[policy]),
            exact_decision_identity=[s for s,h in source_digests.items() if h==digest])
    result=dict(success=True,case=case['id'],sample=sample,records=count,source_sha256=source_hashes,
        source_decision_sha256=source_digests,policies=results,guard=guard,truth_used=False,
        source_GT_alleles_phase_confidence_preserved=True,neural_weights_changed=False,
        wall_seconds=time.monotonic()-begin,code_sha256=sha(__file__))
    save(marker,result);print('REPAIR_COMPLETE',case['id'],{p:r['stats']['changed_vs_current'] for p,r in results.items()},flush=True)
    return result

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--registration',required=True);p.add_argument('--root',required=True);p.add_argument('--case',required=True);a=p.parse_args()
    plan=json.loads(Path(a.registration).read_text());root=Path(a.root);case=dict(next(c for c in plan['cases'] if c['id']==a.case),chrom=plan['chrom'])
    guard=json.loads((root/'GUARD.json').read_text());assert guard['registration_sha256']==sha(a.registration)
    profile=json.loads((root/'profiles'/(case['id']+'.json')).read_text())
    repair(case,profile,guard,root/'cases'/case['id'])

if __name__=='__main__':main()
