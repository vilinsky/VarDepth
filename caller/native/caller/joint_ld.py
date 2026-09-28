"""Joint SNP/indel copying likelihoods for arbitrary regional neural components.

Inputs contain read PL and truth-free candidate metadata. Sample names or
coordinates never select an algorithm branch. Each sample chooses reference
donors from its own SNP likelihoods; automatic cohort evidence remains a
separate dependent expert in the neural output.
"""
from reference_cache import CachedReference
import hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import pysam

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'base'))
from categorical import impute
from panel_fast import load_panel_fast
from state_selection import select_haplotypes
from genetic_map import transitions
from local_alleles import minimal_edit,compatible_edits,apply_edits
from row_patterns import unique_rows

def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()

def collect_components(components):
    """Each component has neural VCF, owned start/end and feature metadata."""
    entries=[];inputs={};header=None;exclusions=set()
    for component in components:
        metadata={}
        left,right=component['start0'],component['end0']
        for path in component['features']:
            p=Path(path)/'metadata.json';inputs[str(p)]=sha(p)
            metadata_file=json.loads(p.read_text())
            exclusions.update(metadata_file.get('base',{}).get('excluded_panel_samples',[]))
            for row in metadata_file['records']:
                if left<=row['start0']<right:metadata[row['record_id']]=row
        p=Path(component['neural']);inputs[str(p)]=sha(p)
        with pysam.VariantFile(p) as source:
            if header is None:header=source.header.copy()
            elif list(header.samples)!=list(source.header.samples):raise ValueError('Component sample order differs')
            for record in source:
                if not left<=record.start<right:continue
                rid=f'{record.chrom}:{record.pos}:{record.ref}:{",".join(record.alts or [])}'
                if rid not in metadata:raise ValueError('Candidate metadata missing: '+rid)
                row=metadata[rid]
                if list(record.alleles)!=row['alleles']:raise ValueError('Candidate allele order mismatch')
                entries.append((record.copy(),row))
    entries.sort(key=lambda item:(item[0].start,item[0].alleles))
    if len({row['record_id'] for _,row in entries})!=len(entries):raise ValueError('Overlapping candidate ownership')
    if len({record.chrom for record,_ in entries})>1:raise ValueError('Joint LD must use one contig')
    return entries,header,inputs,exclusions

def project_donors(entries,panel,chosen,fetch):
    donor=np.full((len(entries),len(chosen)),255,np.uint8)
    for t,(record,row) in enumerate(entries):
        seeds=[(s['start0'],s['ref'],s['alt']) for s in row['seed_variants']]
        if (len(seeds)==1 and len(row['alleles'])==2 and all(len(a)==1 for a in row['alleles'])
                and seeds[0]==(row['start0'],*row['alleles'])
                and row['evidence_start0']==row['start0'] and row['evidence_end0']==row['end0']
                and fetch(row['start0'],row['end0'])==row['alleles'][0]):
            values=panel[seeds[0]][0][chosen] if seeds[0] in panel else np.zeros(len(chosen))
            donor[t]=np.where(values==0,0,np.where(values==1,1,255)).astype(np.uint8)
            continue
        vectors=np.stack([panel[s][0][chosen] if s in panel else np.zeros(len(chosen)) for s in seeds],axis=1)
        patterns,inverse=unique_rows(vectors,return_inverse=True)
        start=min(row['evidence_start0'],row['start0'],min(s[0] for s in seeds))
        end=max(row['evidence_end0'],row['end0'],max(s[0]+len(s[1]) for s in seeds))
        reference=fetch(start,end);prefix=fetch(start,row['start0']);suffix=fetch(row['end0'],end)
        candidates={prefix+allele+suffix:i for i,allele in enumerate(row['alleles'])}
        mapping=np.full(len(patterns),255,np.uint8)
        for i,pattern in enumerate(patterns):
            if np.any((pattern!=0)&(pattern!=1)):continue
            edits=[minimal_edit(s) for s,bit in zip(seeds,pattern) if bit]
            if not compatible_edits(edits):continue
            sequence=apply_edits(reference,start,edits)
            if sequence in candidates:mapping[i]=candidates[sequence]
        donor[t]=mapping[inverse]
    return donor

def run(components,reference,panel_path,genetic_map,chrom,start,end,output,*,exclude_samples=(),states=1024,mutation=.0002,ne=100000,burn=24,iterations=96,chains=2,seed=17):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    entries,header,inputs,component_exclusions=collect_components(components)
    if not entries:return {}
    samples=list(header.samples)
    inputs.update({str(p):sha(p) for p in [reference,panel_path,genetic_map]})
    with CachedReference(reference) as fa:
        fetch=lambda s,e:fa.fetch(chrom,max(0,s),min(e,fa.get_reference_length(chrom))).upper()
        excluded=set(exclude_samples)|set(samples)|component_exclusions
        panel,names,skips=load_panel_fast(panel_path,chrom,start,end,fetch,excluded,50,reference_identity=reference)
        if not names:raise ValueError('Reference panel contains no eligible samples')
        snps=[(v,r) for v,r in entries if len(v.alleles)==2 and len(v.ref)==len(v.alts[0])==1 and (v.start,v.ref,v.alts[0]) in panel]
        nh=len(names)*2
        h=np.stack([panel[(v.start,v.ref,v.alts[0])][0] for v,r in snps]) if snps else np.empty((0,nh))
        positions=np.array([v.start for v,r in snps],np.int64)
        all_positions=np.array([v.start for v,r in entries],np.int64)
        sizes=np.array([len(v.alleles)*(len(v.alleles)+1)//2 for v,r in entries],np.int64)
        offsets=np.concatenate(([0],sizes.cumsum())).astype(np.int64)
        metadata=[dict(record_id=r['record_id'],position=v.start,alleles=list(v.alleles)) for v,r in entries]
        refresh=transitions(all_positions,reference_haplotypes=nh,path=genetic_map,ne=ne,chrom=chrom)
        results={}
        for sample_index,sample in enumerate(samples):
            # Numeric directories keep arbitrary valid sample IDs out of paths.
            dest=out/f'sample-{sample_index:05d}';dest.mkdir(exist_ok=True)
            trial=dest/'posterior';trial.mkdir(exist_ok=True)
            gl=np.array([-np.asarray(v.samples[sample]['PL'])*math.log(10)/10 for v,r in snps]) if snps else np.empty((0,3))
            chosen=select_haplotypes(h,gl,min(states,nh),positions=positions,mode='paired',windows=16)
            donor=project_donors(entries,panel,chosen,fetch)
            local_gl=[-np.asarray(v.samples[sample]['PL'])*math.log(10)/10 for v,r in entries]
            prepared=dict(sample=sample,chrom=chrom,records_metadata=metadata,selected_donors=chosen.tolist(),panel_samples=len(names),panel_skips=skips,
                inputs_sha256=inputs,truth_used=False,excluded_panel_samples=sorted(excluded),mean_unknown_donor_fraction=float((donor==255).mean()),
                settings=dict(states=states,mutation=mutation,ne=ne,burn=burn,iterations=iterations,chains=chains,seed=seed),
                missing_donor_semantics='Unknown donor sequence marginalized uniformly over retained candidate alleles')
            from evidence_trace import enabled, donors
            if enabled():
                prepared['lineage_conditioning_haplotypes']=donors(inputs[str(panel_path)],chrom,names,chosen)
                prepared['lineage_panel_known_fraction']=(donor!=255).mean(axis=1).tolist()
            (dest/'PREPARED.json').write_text(json.dumps(prepared,separators=(',',':'))+'\n')
            begun=time.monotonic()
            posterior=impute(donor,local_gl,refresh,mutation=mutation,burn=burn,iterations=iterations,chains=chains,seed=seed)
            np.savez_compressed(trial/'posterior.npz',gp=np.concatenate(posterior),offsets=offsets)
            (trial/'COMPLETE.json').write_text(json.dumps(dict(seconds=time.monotonic()-begun,posterior_sha256=sha(trial/'posterior.npz'),prepared_sha256=sha(dest/'PREPARED.json'),truth_used=False),indent=2)+'\n')
            results[sample]=str(trial)
    return results
