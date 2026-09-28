"""Learned, dependent-expert stacking; no truth or locus identity at inference.

All experts may reuse the same molecules. Their posteriors are combined by an
arithmetic mixture, never by a product asserting independent evidence. Missing
experts are explicitly masked; the neural source already includes automatic
leave-self-out cohort context. A singleton is a regular input.
"""
import json
from functools import lru_cache
import math
from pathlib import Path
import numpy as np

EXPERTS = ('neural', 'external_hmm', 'snp_scaffold', 'anchored', 'reads',
           'anchored_weak', 'reads_weak', 'joint_ld', 'spanning_anchor', 'spanning_weak')
MAX_ALLELES = 48
MAX_GP = MAX_ALLELES * (MAX_ALLELES + 1) // 2
VERSION = '0.5.11-graph-whole-high-residual-trust'
EXPERT_IDENTITY=np.eye(len(EXPERTS),dtype=np.float32)


@lru_cache(maxsize=24)
def pairs(n):
    return tuple((a, b) for b in range(n) for a in range(b + 1))


def catalog_signature(record):
    start=record['start0'];alleles=list(record['alleles'])
    while min(map(len,alleles))>1 and len({a[-1] for a in alleles})==1:
        alleles=[a[:-1] for a in alleles]
    while min(map(len,alleles))>1 and len({a[0] for a in alleles})==1:
        start+=1;alleles=[a[1:] for a in alleles]
    return start,tuple(alleles)


def probability(values, ng):
    if ng==3 and len(values)==3:
        a,b,c=map(float,values);total=(a+b)+c
        if min(a,b,c)<0 or not all(map(math.isfinite,(a,b,c))) or total<=0:
            raise ValueError('Invalid expert distribution')
        return np.array((a/total,b/total,c/total),np.float64)
    x = np.zeros(ng, np.float64)
    v = np.asarray(values, np.float64)
    if len(v) > ng or not np.isfinite(v).all() or np.any(v < 0) or v.sum() <= 0:
        raise ValueError('Invalid expert distribution')
    x[:len(v)] = v / v.sum()
    return x


def likelihood_posterior(pl, allele_frequency, prior_power=1.):
    """Temper a frequency prior, not independent repetitions of read evidence."""
    af = np.maximum(np.asarray(allele_frequency, float), .001)
    af /= af.sum()
    prior = np.array([af[a] * af[b] * (1 if a == b else 2)
                      for a, b in pairs(len(af))])
    lp = -np.asarray(pl, float) * math.log(10) / 10 + prior_power * np.log(prior)
    p = np.exp(lp - lp.max())
    return p / p.sum()


def scaffold_data(path):
    if not path:
        return {}, {}
    meta = json.loads(Path(path).with_suffix('.json').read_text())
    chrom = meta['arguments']['region'].split(':')[0]
    with np.load(path, allow_pickle=False) as z:
        samples = {str(s): i for i, s in enumerate(z['samples'])}
        gp = z['gp']  # NPZ entries decompress on access; load exactly once.
        rows = {(chrom, int(p), str(r), str(a)): gp[:, i].copy()
                for i, (p, r, a) in enumerate(zip(z['positions'], z['ref'], z['alt']))}
    return rows, samples


def joint_ld_data(directory, expected_sample=None):
    if not directory:
        return {}
    directory = Path(directory)
    meta = json.loads((directory.parent / 'PREPARED.json').read_text())
    if expected_sample is not None and meta['sample'] != expected_sample:
        raise ValueError('Joint LD sample identity mismatch')
    with np.load(directory / 'posterior.npz', allow_pickle=False) as z:
        gp, offsets = z['gp'], z['offsets']
        if len(offsets) != len(meta['records_metadata']) + 1:
            raise ValueError('Joint LD metadata/probability count mismatch')
        sizes = np.array([len(r['alleles']) * (len(r['alleles']) + 1) // 2 for r in meta['records_metadata']])
        if offsets[0] != 0 or offsets[-1] != len(gp) or not np.array_equal(np.diff(offsets), sizes):
            raise ValueError('Joint LD genotype vector/order mismatch')
        if not np.isfinite(gp).all() or np.any(gp < 0):
            raise ValueError('Invalid joint LD probabilities')
        return {r['record_id']: gp[offsets[t]:offsets[t+1]].copy()
                for t,r in enumerate(meta['records_metadata'])}


def parent_bundle(record, sample, refined=None, scaffold=None, joint_ld=None, spanning=None):
    """Construct expert distributions and permutation-neutral local features."""
    call = record.samples[sample]
    alleles = refined['alleles'] if refined else list(record.alleles)
    n = len(alleles); ng = n * (n + 1) // 2
    if not 2 <= n <= MAX_ALLELES:
        raise ValueError('Unsupported allele count')
    order = pairs(n)
    panel_af = list(record.info['PANEL_AF']) if 'PANEL_AF' in record.info else [None] * (len(record.alleles) - 1)
    panel_af = [max(.001, float(v or 0)) for v in panel_af]
    allele_frequency = [max(.001, 1 - sum(panel_af))] + panel_af + [.001] * (n - len(record.alleles))
    dp = float(call.get('DP', 0) or 0)
    cs = float(call.get('CS', 0) or 0)
    indel = any(len(a) != len(alleles[0]) for a in alleles)
    ps = np.zeros((len(EXPERTS), MAX_GP), np.float32)
    available = np.zeros(len(EXPERTS), bool)
    ps[0, :ng] = probability(call['GP'], ng); available[0] = True
    ps[1, :ng] = probability(call.get('BGP', call['GP']), ng); available[1] = True
    if joint_ld is not None:
        ps[7, :ng] = probability(joint_ld, ng); available[7] = True
    if scaffold is not None:
        if n != 2 or indel or len(alleles[0]) != 1:
            raise ValueError('SNP scaffold must have identical biallelic alleles')
        ps[2, :ng] = probability(scaffold, ng); available[2] = True
    expert = refined['samples'][sample] if refined else None
    full_expert = spanning['samples'][sample] if spanning else None
    if spanning:
        comparison = refined or dict(start0=record.start, alleles=list(record.alleles))
        prefix = dict(start0=comparison['start0'], alleles=comparison['alleles'][:len(spanning['alleles'])])
        if len(comparison['alleles']) < len(spanning['alleles']) or catalog_signature(spanning) != catalog_signature(prefix):
            raise ValueError('Original spanning alleles must remain an ordered catalog prefix')
    if full_expert and full_expert['dp'] > 0:
        ps[8, :ng] = probability(full_expert['gp'], ng); available[8] = True
        full_n = len(spanning['alleles'])
        full_weak = likelihood_posterior(full_expert['pl'], allele_frequency[:full_n], .25)
        ps[9, :len(full_weak)] = full_weak; available[9] = True
    if expert and expert['dp'] > 0:
        ps[3, :ng] = probability(expert['gp'], ng)
        # Hard confidence/support gates discarded even a correct likelihood.
        # Availability means evidence exists. Its reliability is learned below;
        # old trusted status remains an input feature, never a truth-based rule.
        available[3] = True
        ps[5, :ng] = likelihood_posterior(expert['pl'], allele_frequency, .25)
        available[5] = True
    # The baseline PL vector has no entries for new alleles. Do not fabricate
    # evidence for them. Expanded anchored likelihoods are a separate source.
    pl = call.get('PL')
    read_best = 0
    read_lp = np.full(ng, -60., np.float64)
    if pl is not None and len(pl):
        lp = -np.asarray(pl, float) * math.log(10) / 10
        read_lp[:len(lp)] = lp - lp.max()
        af = list(record.info['PANEL_AF']) if 'PANEL_AF' in record.info else []
        if len(af) != len(record.alleles) - 1:
            af = [None] * (len(record.alleles) - 1)
        af = [max(.001, float(v)) if v is not None else .001 for v in af]
        af = [max(.001, 1 - sum(af))] + af
        af = np.array(af); af /= af.sum()
        prior = np.array([af[a] * af[b] * (1 if a == b else 2)
                          for a, b in pairs(len(af))])
        lp = lp + np.log(prior)
        rp = np.exp(lp - lp.max()); rp /= rp.sum()
        ps[4, :len(rp)] = rp; available[4] = True
        if dp > 0:
            ps[6, :len(rp)] = likelihood_posterior(pl, allele_frequency[:len(record.alleles)], .25)
            available[6] = True
        read_best = int(np.argmax(rp))
    ad = np.asarray(call.get('AD', [0] * len(record.alleles)), float)
    adsum = float(ad.sum())
    ads = np.sort(ad)[::-1]
    minor = float(ads[1] / max(1., adsum)) if len(ads) > 1 else 0.
    aa = np.asarray(expert['exact_ad'], float) if expert else np.zeros(n)
    aorder = sorted(aa, reverse=True)
    adp = float(expert['dp']) if expert else 0.
    # None of these features identifies a person, chromosome, coordinate, seed,
    # nominal coverage, benchmark status, or known pathogenic allele.
    context = [math.log1p(dp) / math.log(101), cs, float(indel),
               math.log1p(len(alleles[0])) / math.log(101),
               math.log(n) / math.log(24), minor,
               min(1., adsum / max(1., dp)),
               math.log1p(max(abs(len(a) - len(alleles[0])) for a in alleles)) / math.log(101),
               min(2., adp / max(1., dp)),
               float(expert['trusted']) if expert else 0.,
               float(expert['support_fraction']) if expert else 0.,
               aorder[1] / max(1., adp) if len(aorder) > 1 else 0.,
               float(refined['added_alleles']) / 24 if refined else 0.,
               float('SOURCE' in record.info and record.info['SOURCE'] == 'READS')]
    winners = ps[:, :ng].argmax(1)
    sorted_probabilities = np.sort(ps[:, :ng], axis=1)
    available_count = max(1, available.sum())
    prepared = {}
    for full, ex in [(False, expert), (True, full_expert)]:
        aa = np.asarray(ex['exact_ad'], float) if ex else np.zeros(n)
        adp = float(ex['dp']) if ex else 0.
        local_context = list(context)
        local_context[8] = min(2., adp / max(1., dp))
        local_context[9] = float(ex['trusted']) if ex else 0.
        local_context[10] = float(ex['support_fraction']) if ex else 0.
        aaorder = np.sort(aa)[::-1]
        local_context[11] = aaorder[1] / max(1., adp) if len(aaorder) > 1 else 0.
        apl = np.asarray(ex['pl'], float) if ex else np.zeros(ng)
        sorted_pl = np.sort(apl)
        prepared[full] = (ex, aa, adp, local_context, apl, sorted_pl, {})
    entropy_values=-(ps[:,:ng]*np.log(np.maximum(ps[:,:ng],1e-30))).sum(axis=1)
    agreement_counts=np.bincount(winners[available],minlength=ng)
    features = np.empty((len(EXPERTS),42),np.float32)
    features[:,-len(EXPERTS):]=EXPERT_IDENTITY
    for e, p in enumerate(ps[:, :ng]):
        ex, aa, adp, local_context, apl, sorted_pl, cache = prepared[e in (8, 9)]
        top = sorted_probabilities[e, -2:]
        confidence = -math.log(max(1e-9, 1 - float(top[-1]))) / 20
        entropy = float(entropy_values[e]) / math.log(ng)
        agreement = float(agreement_counts[winners[e]]) / available_count
        local = [confidence, entropy, float(p[0]), float(top[-1] - top[-2]), agreement,
                 float(winners[e] == winners[0]), float(winners[e] == read_best),
                 float(read_lp[winners[e]]) / 60, float(winners[e] != 0)]
        winner = int(winners[e])
        if winner not in cache:
            a, b = order[winner]
            if ex is None:
                cache[winner]=[0.,0.,0.,0.,0.,0.,float(a!=b),float(a>=len(record.alleles) or b>=len(record.alleles)),0.]
        if winner not in cache:
            a, b = order[winner]
            selected = sorted(set((a, b)))
            support = sum(aa[j] for j in selected) / max(1., adp)
            minimum = min(aa[j] for j in selected) / max(1., adp)
            balance = min(aa[a], aa[b]) / max(1., aa[a] + aa[b]) if a != b else 0.
            minimum_count = min(aa[j] for j in selected)
            if ex and 'exact_genotype_counts' in ex:
                counts = ex['exact_genotype_counts']
                if len(counts) != ng: raise ValueError('Diploid support genotype order mismatch')
                only_a, only_b, both = counts[winner]
                if min(only_a, only_b, both) < 0 or only_a + only_b + both > adp: raise ValueError('Invalid distinct-molecule genotype support')
                support = (only_a + only_b + both) / max(1., adp)
                minimum_count = only_a if a == b else min(only_a, only_b)
                minimum = minimum_count / max(1., adp)
                balance = min(only_a, only_b) / max(1., only_a + only_b) if a != b else 0.
            chosen_pl = apl[winner] if ex else 0.
            cache[winner] = [support, minimum, balance,
                    math.log1p(minimum_count) / math.log(101),
                    min(10., chosen_pl / 60),
                    min(10., (sorted_pl[1] - sorted_pl[0]) / 60) if ex else 0.,
                    float(a != b), float(any(j >= len(record.alleles) for j in selected)),
                    float(winner == int(apl.argmin())) if ex else 0.]
        features[e,:32]=local_context + local + cache[winner]
    return {'features': features, 'probabilities': ps,
            'available': available, 'n_genotypes': ng, 'indel': indel,
            'alleles': alleles, 'start0': refined['start0'] if refined else record.start}


def model_class():
    import torch
    from torch import nn
    class TrustNet(nn.Module):
        def __init__(self, inputs=42, hidden=48, architecture='plain'):
            super().__init__()
            self.inputs = inputs; self.hidden = hidden
            self.architecture = architecture
            self.scorer = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(),
                                        nn.Linear(hidden, 1))
            nn.init.zeros_(self.scorer[-1].weight)
            nn.init.zeros_(self.scorer[-1].bias)
            self.bias = nn.Parameter(torch.tensor([1., 0., 0., -1., -1., -1., -1., 0., -1., -1.]))
            if architecture in ('residual_stratified', 'indel_unbounded'):
                self.indel_scorer = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, 1))
                nn.init.zeros_(self.indel_scorer[-1].weight)
                nn.init.zeros_(self.indel_scorer[-1].bias)
                nn.init.zeros_(self.bias)
            if architecture == 'indel_unbounded':
                self.indel_choice = nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(),
                    nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
                nn.init.zeros_(self.indel_choice[-1].weight)
                nn.init.zeros_(self.indel_choice[-1].bias)
                self.indel_bias = nn.Parameter(torch.zeros(len(EXPERTS)))
        def forward(self, x, probabilities, available, n_genotypes):
            score = self.scorer(x).squeeze(-1)
            if self.architecture in ('residual_stratified', 'indel_unbounded'):
                is_indel = x[:, 0, 2] > .5
                score = torch.where(is_indel[:, None], self.indel_scorer(x).squeeze(-1), score)
                dp = torch.expm1(x[:, 0, 0] * math.log(101)).clamp_min(0)
                sw = .9 * torch.exp(-dp / 4) * (1 - x[:, 0, 1].clamp(0, .5)) * available[:, 2]
                aw = .8 * ((dp - 15) / 15).clamp(0, 1) * available[:, 3] * (x[:, 0, 11] >= .25)
                lw = .9 * torch.exp(-dp / 8) * available[:, 7]
                anchor = torch.stack([(1 - sw) * (1 - aw), torch.zeros_like(sw),
                                      sw * (1 - aw), aw, torch.zeros_like(sw),
                                      torch.zeros_like(sw), torch.zeros_like(sw)], dim=1)
                anchor = torch.cat([anchor * (1 - lw[:, None]), lw[:, None], torch.zeros_like(lw[:, None]), torch.zeros_like(lw[:, None])], dim=1)
                # The previous usable policy is the initialization. The learned
                # residual can change its relative odds by exp(+/-6); these
                # are trained weights, not a fixed output switch by coverage.
                logits = anchor.clamp_min(.005).log() + 6 * torch.tanh(score) + self.bias
                if self.architecture == 'indel_unbounded':
                    # The indel branch learns log-odds directly. It has no
                    # fixed preference for the old neural or spanning source.
                    # SNV computations retain the frozen r03 branch exactly.
                    choices = self.indel_choice(x).squeeze(-1) + self.indel_bias
                    logits = torch.where(is_indel[:, None], choices, logits)
            else:
                logits = score + self.bias
            logits = logits.masked_fill(~available, -1e4)
            weights = torch.softmax(logits, dim=-1)
            p = (weights[:, :, None] * probabilities).sum(1)
            mask = torch.arange(p.shape[1], device=p.device)[None, :] < n_genotypes[:, None]
            # A finite loss for valid, previously absent genotypes; no mass on
            # padded genotypes. This floor is not a discovery mechanism.
            p = p.clamp_min(1e-12) * mask
            # Preserve the trained full-width reduction tree. Compact zero
            # padding saves expert operations, but changing this float32 sum
            # can change high-confidence GQ by several Phred units.
            denominator = torch.nn.functional.pad(p, (0, MAX_GP-p.shape[1])).sum(1, keepdim=True) if p.shape[1]<MAX_GP else p.sum(1, keepdim=True)
            p = p / denominator
            return p, weights
    return TrustNet


class Predictor:
    def __init__(self, checkpoint, disabled=()):
        import torch
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.get('version') != VERSION:
            raise ValueError('Unrecognized trained trust model')
        architecture = state['config'].get('architecture')
        self.residual = architecture in ('genotype_residual', 'high_snv_residual', 'high_snv_read_guard')
        if architecture in ('high_snv_residual', 'high_snv_read_guard'):
            from high_snv_model import model_class as high_model_class
            self.model = high_model_class()(**state['config'])
        else:
            self.model = (residual_model_class() if self.residual else model_class())(**state['config'])
        self.model.load_state_dict(state['state']); self.model.eval()
        self.temperature = float(state.get('temperature', 1.))
        self.disabled = [EXPERTS.index(e) for e in disabled]
    def __call__(self, rows):
        import torch
        import os
        if not rows:
            return [], []
        x = torch.from_numpy(np.stack([r['features'] for r in rows]))
        width=max(r['n_genotypes'] for r in rows) if os.environ.get('VARDEPTH_COMPACT_NEURAL')=='1' else MAX_GP
        probabilities=np.zeros((len(rows),len(EXPERTS),width),np.float32)
        genotype_features=np.zeros((len(rows),width,20),np.float32) if self.residual else None
        for i,r in enumerate(rows):
            ng=r['n_genotypes'];probabilities[i,:,:ng]=r['probabilities'][:,:ng]
            if self.residual:genotype_features[i,:ng]=r['genotype_features'][:ng]
        p = torch.from_numpy(probabilities)
        a = torch.from_numpy(np.stack([r['available'] for r in rows]))
        n = torch.tensor([r['n_genotypes'] for r in rows])
        if self.disabled:
            a[:, self.disabled] = False
        if not a.any(1).all():
            raise ValueError('No usable experts after ablation')
        with torch.inference_mode():
            gp, weights = self.model(x, p, a, n, torch.from_numpy(genotype_features)) if self.residual else self.model(x, p, a, n)
            if self.temperature != 1:
                mask = torch.arange(gp.shape[1])[None, :] < n[:, None]
                gp = gp.pow(1 / self.temperature) * mask
                denominator = torch.nn.functional.pad(gp, (0, MAX_GP-gp.shape[1])).sum(1, keepdim=True) if gp.shape[1]<MAX_GP else gp.sum(1, keepdim=True)
                gp /= denominator
        probabilities=gp.numpy()
        if width<MAX_GP:probabilities=np.pad(probabilities,((0,0),(0,MAX_GP-width)))
        return probabilities, weights.numpy()


def genotype_evidence(record, sample, item, refined, spanning):
    """Permutation-neutral, genotype-specific read evidence; no locus identifiers."""
    n=len(item['alleles']);ng=item['n_genotypes'];order=np.asarray(pairs(n),dtype=np.int64)
    a,b=order[:,0],order[:,1];heter=a!=b;length=np.array([len(x) for x in item['alleles']])
    result=np.zeros((MAX_GP,20),np.float32)
    result[:ng,0]=heter
    result[:ng,1]=(a>=len(record.alleles))|(b>=len(record.alleles))
    result[:ng,2]=np.log1p(np.maximum(abs(length[a]-length[0]),abs(length[b]-length[0])))/math.log(101)
    result[:ng,3]=np.log1p(abs(length[a]-length[b]))/math.log(101)
    for offset,source in [(4,refined),(11,spanning)]:
        if not source:continue
        ex=source['samples'][sample];dp=float(ex['dp']);source_n=len(source['alleles']);source_ng=source_n*(source_n+1)//2
        if dp<=0:continue
        valid=(a<source_n)&(b<source_n);aa=np.zeros(n);aa[:source_n]=ex['exact_ad']
        if 'exact_genotype_counts' in ex:
            counts=np.asarray(ex['exact_genotype_counts'],float)
            if counts.shape!=(source_ng,3):raise ValueError('Invalid genotype evidence dimensions')
            padded=np.zeros((ng,3));padded[:source_ng]=counts
            only_a,only_b,both=padded.T
        else:
            only_a=aa[a];only_b=np.where(heter,aa[b],0.);both=np.zeros(ng)
        if np.any(only_a+only_b+both>dp) or np.any(only_a<0) or np.any(only_b<0):raise ValueError('Genotype evidence double-counts molecules')
        minimum=np.where(heter,np.minimum(only_a,only_b),only_a)
        result[:ng,offset]=valid
        result[:ng,offset+1]=(only_a+only_b+both)/dp
        result[:ng,offset+2]=minimum/dp
        result[:ng,offset+3]=np.where(heter,np.minimum(only_a,only_b)/np.maximum(1,only_a+only_b),0)
        result[:ng,offset+4]=both/dp
        result[:ng,offset+5]=np.log1p(minimum)/math.log(101)
        pl=np.asarray(ex['pl'],float)
        if len(pl)!=source_ng:raise ValueError('Read likelihood dimensions disagree with catalog')
        result[:source_ng,offset+6]=np.minimum(10,pl/60)
    pl=record.samples[sample].get('PL')
    if pl is not None:
        size=len(pl);result[:size,18]=1;result[:size,19]=np.minimum(10,np.asarray(pl,float)/60)
    return result


def bundle(record, sample, refined=None, scaffold=None, joint_ld=None, spanning=None, *, compact=False):
    if refined is None and spanning is None and len(record.alleles)==2 and all(len(a)==1 for a in record.alleles):
        from trust_cpu import bundle as snv_bundle
        result=snv_bundle(record,sample,scaffold,joint_ld,3 if compact else MAX_GP)
        if result is not None:return result
    item=parent_bundle(record,sample,refined,scaffold,joint_ld,spanning)
    item['genotype_features']=genotype_evidence(record,sample,item,refined,spanning)
    if compact:
        item['probabilities']=item['probabilities'][:,:item['n_genotypes']].copy()
        item['genotype_features']=item['genotype_features'][:item['n_genotypes']].copy()
    return item


def residual_model_class():
    import torch
    from torch import nn
    class GenotypeResidual(nn.Module):
        def __init__(self,inputs=42,hidden=48,architecture='genotype_residual'):
            super().__init__()
            if architecture!='genotype_residual':raise ValueError('Wrong residual architecture')
            self.parent=model_class()(inputs=inputs,hidden=hidden,architecture='indel_unbounded')
            for parameter in self.parent.parameters():parameter.requires_grad_(False)
            self.residual=nn.Sequential(nn.Linear(64,48),nn.Tanh(),nn.Dropout(.05),nn.Linear(48,32),nn.Tanh(),nn.Linear(32,1))
            nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
            import copy
            self.low_residual=copy.deepcopy(self.residual)
            for parameter in self.low_residual.parameters():parameter.requires_grad_(False)
        def forward(self,x,probabilities,available,n_genotypes,genotype_features):
            p,weights=self.parent(x,probabilities,available,n_genotypes)
            indel=x[:,0,2]>.5
            valid=(torch.arange(p.shape[1],device=p.device)[None,:]<n_genotypes[:,None])&indel[:,None]
            rows,columns=valid.nonzero(as_tuple=True)
            if len(rows)==0:return p,weights
            source=probabilities[rows,:,columns]
            inputs=torch.cat([x[rows,0,:14],source,source.clamp_min(1e-12).log()/30,
                              available[rows].float(),genotype_features[rows,columns]],dim=1)
            delta=torch.zeros_like(p)
            self.low_residual.eval()
            with torch.no_grad():low=self.low_residual(inputs).squeeze(-1)
            high=self.residual(inputs).squeeze(-1)
            high_depth=x[rows,0,0]>=math.log1p(19.5)/math.log(101)
            delta[rows,columns]=4*torch.tanh(torch.where(high_depth,high,low))
            updated=p*delta.exp()
            denominator = torch.nn.functional.pad(updated, (0, MAX_GP-updated.shape[1])).sum(1, keepdim=True) if updated.shape[1]<MAX_GP else updated.sum(1,keepdim=True)
            updated=updated/denominator
            # Keep pure-SNV computation exactly equal to the frozen parent.
            return torch.where(indel[:,None],updated,p),weights
    return GenotypeResidual
