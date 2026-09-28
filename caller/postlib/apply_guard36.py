"""Apply the frozen v36 research guard to matching single-sample source VCFs.

No truth, training labels, or requested nominal depth is used. Input VCFs must
cover the same chromosome and have identical ordered alleles. This is a safety
postprocessor for the frozen VarDepth sources, not a standalone variant caller.
"""
import argparse,collections,math,time,json
from pathlib import Path
import pysam
from profile_depth36 import quantile
from repair36 import repair,save,sha

def make_profile(native,sample,chrom,length,guard):
    start=time.monotonic();width=guard['window_bp'];hist=collections.Counter();windows=collections.defaultdict(collections.Counter);stats=collections.Counter();last=None
    with pysam.VariantFile(native) as f:
        assert list(f.header.samples)==[sample]
        for v in f:
            assert v.chrom==chrom,'Supply one chromosome per invocation'
            stats['records']+=1
            if len(v.alleles)!=2 or any(len(a)!=1 or a not in 'ACGT' for a in v.alleles) or v.pos==last:continue
            dp=v.samples[sample].get('DP')
            if dp is None:stats['missing_DP']+=1;continue
            assert isinstance(dp,int) and dp>=0
            assert 0<=v.start<length
            last=v.pos;hist[dp]+=1;windows[v.start//width][dp]+=1;stats['anchors']+=1
    rows=[]
    for i in range(math.ceil(length/width)):
        h=windows[i];rows.append(dict(index=i,start=i*width,end=min((i+1)*width,length),anchors=sum(h.values()),q25=quantile(h,.25),median_DP=quantile(h,.5),q75=quantile(h,.75)))
    return dict(sample=sample,source=str(native),source_sha256=sha(native),stats=dict(stats),median_DP=quantile(hist,.5),windows=rows,truth_used=False,wall_seconds=time.monotonic()-start)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',required=True);p.add_argument('--current',required=True);p.add_argument('--guard',required=True);p.add_argument('--sample',required=True);p.add_argument('--chrom',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    native=Path(a.native).resolve();current=Path(a.current).resolve();out=Path(a.out).resolve();guard=json.loads(Path(a.guard).read_text())
    assert native!=current and not out.exists(),'Output must be a new directory'
    assert guard['success'] and guard['lower']==15.5 and guard['upper']==62 and guard['window_bp']==4000000 and guard['minimum_anchors']==100,'Use the frozen validated v36 guard'
    with pysam.VariantFile(native) as f:length=f.header.contigs[a.chrom].length
    assert length and length>0,'Native VCF must declare chromosome length'
    prof=make_profile(native,a.sample,a.chrom,length,guard)
    save(out/'PROFILE.json',prof)
    case=dict(id=a.sample+'-'+a.chrom,sample=a.sample,chrom=a.chrom,native=str(native),current=str(current))
    result=repair(case,prof,guard,out,policies=('coverage-domain-guard',))
    save(out/'RUN.json',dict(success=True,guard_sha256=sha(a.guard),cli_sha256=sha(__file__),selected_policy='coverage-domain-guard',output=result['policies']['coverage-domain-guard']['output']))
if __name__=='__main__':main()
