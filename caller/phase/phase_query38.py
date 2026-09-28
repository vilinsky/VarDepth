"""Read a cis/trans answer from an explicit phase block; unknown is valid."""
import argparse,json
import pysam

def relationship(first,first_alt,second,second_alt,sample):
    if first.contig!=second.contig:return dict(result='unknown',reason='different_chromosomes')
    a=first.samples[sample];b=second.samples[sample]
    for record,call,allele in [(first,a,first_alt),(second,b,second_alt)]:
        if allele<1 or allele>=len(record.alleles):raise ValueError('ALT index outside record')
        gt=call.get('GT')
        if not gt or len(gt)!=2 or None in gt or gt.count(allele)!=1:
            return dict(result='unknown',reason='target_allele_not_heterozygous')
        if not call.phased or call.get('PS') is None:return dict(result='unknown',reason='unphased_variant')
    if a['PS']!=b['PS']:return dict(result='unknown',reason='different_phase_blocks')
    result='cis' if a['GT'].index(first_alt)==b['GT'].index(second_alt) else 'trans'
    errors=[c.get('VPE') for c in [a,b]]
    return dict(result=result,phase_set=a['PS'],model_block_error=max(errors) if None not in errors else None,
        empirically_calibrated=False,parental_origin_known=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--vcf',required=True);p.add_argument('--sample',required=True)
    p.add_argument('--variant',action='append',required=True,help='CHROM:POS:REF:ALT; provide exactly twice')
    a=p.parse_args()
    if len(a.variant)!=2:p.error('Exactly two variants are required')
    found=[]
    with pysam.VariantFile(a.vcf) as f:
        for text in a.variant:
            c,pos,ref,alt=text.split(':');pos=int(pos)
            rows=[(r.copy(),r.alleles.index(alt)) for r in f.fetch(c,pos-1,pos) if r.pos==pos and r.ref==ref and alt in r.alts]
            if len(rows)!=1:raise ValueError('Variant absent or ambiguous: '+text)
            found.append(rows[0])
    print(json.dumps(relationship(found[0][0],found[0][1],found[1][0],found[1][1],a.sample),indent=2))

if __name__=='__main__':main()
