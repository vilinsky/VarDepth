"""Numerical experimental features from observed evidence inventories.

Direct molecule overlap and conditioning-set overlap are descriptive, not
correlation estimates. Catalog selection, reference/model training ancestry and
non-local HMM influence are NOT made independent by this representation.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from evidence_trace import read, sidecar, VERSION

NAMES = ['direct_inventory_known', 'log_direct_molecules', 'window_inventory_known',
         'log_window_molecules', 'local_reused_at_other_catalog_loci_fraction',
         'panel_conditioning_known', 'log_conditioning_haplotypes',
         'cohort_selected_donors', 'cohort_effective_donors', 'cohort_weight_sum',
         'cohort_flank_count', 'target_local_in_cohort_flanks_fraction',
         'joint_panel_projection_known_fraction', 'catalog_shared_across_people',
         'library_namespace_complete', 'direct_kind_whole_read_uniform']
for _peer in range(10):
    NAMES.extend([f'direct_jaccard_peer{_peer}', f'direct_overlap_known_peer{_peer}',
                  f'conditioning_jaccard_peer{_peer}', f'conditioning_overlap_known_peer{_peer}'])


def overlap(left, right):
    if left is None or right is None:
        return 0., 0.
    union = len(left) + len(right) - len(left & right)
    # Empty inventories are known, but have no positive overlap evidence.
    return (len(left & right) / union if union else 0.), 1.


def log_count(values):
    return math.log1p(len(values)) / math.log(1001) if values is not None else 0.


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class FeatureWindow:
    def __init__(self, path):
        self.path = Path(path)
        self.trace = read(self.path / 'evidence-trace.json.gz')
        if self.trace['version'] != VERSION:
            raise ValueError('Trace schema mismatch')
        self.meta = json.loads((self.path / 'metadata.json').read_text())
        self.pool = {k: frozenset(v) for k, v in self.trace['inventories'].items()}
        self.rows = {(r['sample'], r['record_id']): r for r in self.trace['rows']}
        if len(self.rows) != len(self.trace['rows']):
            raise ValueError('Duplicate lineage row')
        self.by_index = {(r['sample'], r['record_index']): r for r in self.trace['rows']}
        self.reuse = {}
        for sample in self.meta['samples']:
            counter = Counter()
            for (person, _), row in self.rows.items():
                if person == sample:
                    counter.update(self.pool[row['local_molecules']])
            self.reuse[sample] = counter

    def local(self, sample, index):
        return self.pool[self.by_index[sample, index]['local_molecules']]

    def cohort(self, sample, index, local):
        cohort = self.trace['cohort']
        if cohort is None:
            return [0.] * 5
        target = cohort['samples'].index(sample)
        selected = cohort['donor_indices'][target][index]
        weights = cohort['donor_weights'][target][index]
        active = [(int(i), float(w)) for i, w in zip(selected, weights) if i >= 0 and w > 0]
        if any(cohort['samples'][i] == sample for i, _ in active):
            raise ValueError('Self cohort donor')
        total = sum(w for _, w in active)
        effective = total * total / sum(w*w for _, w in active) if total else 0.
        flanks = cohort['flank_indices'].get(str(index), [])
        target_flanks = set().union(*(self.local(sample, i) for i in flanks)) if flanks else set()
        reuse = len(local & target_flanks) / len(local) if local else 0.
        return [len(active) / 8, effective / 8, min(total, 8.) / 8,
                len(flanks) / 128, reuse]


class LineageFeatures:
    def __init__(self, args):
        component_path = Path(args.neural).parent / 'COMPONENTS.json'
        component = json.loads(component_path.read_text())
        self.inputs = {str(component_path): sha(component_path)}
        self.windows = []
        self.lookup = {}
        for path in component['features']:
            window = FeatureWindow(path)
            self.windows.append(window)
            owned = component['ownership'][str(Path(path).resolve())]
            for row in window.meta['records']:
                if row['chrom'] == owned['chrom'] and owned['start0'] <= row['start0'] < owned['end0']:
                    for sample in window.meta['samples']:
                        key = (sample, row['record_id'])
                        if key in self.lookup:
                            raise ValueError('Overlapping feature ownership')
                        self.lookup[key] = window
            for name in ['evidence-trace.json.gz', 'metadata.json']:
                self.inputs[str(Path(path) / name)] = sha(Path(path) / name)
        self.scaffold = read(sidecar(args.scaffold)) if args.scaffold else None
        if self.scaffold:
            self.inputs[str(sidecar(args.scaffold))] = sha(sidecar(args.scaffold))
            self.scaffold_pool = {k: frozenset(v) for k,v in self.scaffold['inventories'].items()}
        self.joint = {}
        for entry in args.joint_ld:
            sample, directory = entry.split('=', 1)
            path = Path(directory).parent / 'PREPARED.json'
            data = json.loads(path.read_text())
            self.inputs[str(path)] = sha(path)
            if 'lineage_conditioning_haplotypes' not in data:
                raise ValueError('Joint LD lacks observed conditioning trace')
            joined = set()
            for record in data['records_metadata']:
                window = self.lookup.get((sample, record['record_id']))
                # Joint LD can span several components. Read their metadata and
                # traces rather than assuming the current component is complete.
                if window is None:
                    for input_path in data['inputs_sha256']:
                        if Path(input_path).name != 'neural.vcf.gz':
                            continue
                        other_path = Path(input_path).parent/'COMPONENTS.json'
                        other = json.loads(other_path.read_text())
                        self.inputs[str(other_path)] = sha(other_path)
                        for feature_path in other['features']:
                            owned = other['ownership'][str(Path(feature_path).resolve())]
                            if not owned['start0'] <= record['position'] < owned['end0']:
                                continue
                            candidate = next((w for w in self.windows if w.path == Path(feature_path)), None)
                            if candidate is None:
                                candidate = FeatureWindow(feature_path)
                                self.windows.append(candidate)
                                for name in ['metadata.json', 'evidence-trace.json.gz']:
                                    self.inputs[str(candidate.path/name)] = sha(candidate.path/name)
                            if (sample, record['record_id']) in candidate.rows:
                                if window is not None and candidate is not window:
                                    raise ValueError('Ambiguous joint PL ancestry')
                                window = candidate
                if window is None:
                    raise ValueError('Missing joint PL ancestry')
                row = window.rows[sample, record['record_id']]
                joined.update(window.pool[row['local_molecules']])
            self.joint[sample] = dict(window=frozenset(joined),
                panel=frozenset(data['lineage_conditioning_haplotypes']),
                known={r['record_id']: v for r,v in zip(data['records_metadata'], data['lineage_panel_known_fraction'])})

    def row(self, original, sample, refinement, spanning, available):
        rid = f'{original.chrom}:{original.pos}:{original.ref}:{",".join(original.alts or [])}'
        window = self.lookup[sample, rid]
        record = window.rows[sample, rid]
        info = window.trace['windows'][sample]
        pool = window.pool
        local = pool[record['local_molecules']]
        neural = pool[record['neural_molecules']]
        upstream = pool[info['likelihood_molecules']]
        panel = pool[info['conditioning_haplotypes']]
        direct = [None] * 10
        context = [None] * 10
        panels = [None] * 10
        direct[0], direct[1], direct[4], direct[6] = neural, local, local, local
        context[0], context[1], context[4], context[6] = upstream, upstream, local, local
        # Neural receives panel-derived features. Read experts also use panel
        # allele frequencies, but that is not a conditioning-haplotype inventory.
        panels[0] = panels[1] = panel
        sc = self.scaffold['samples'].get(sample) if self.scaffold else None
        if available[2]:
            if sc is None or rid not in sc['local']:
                raise ValueError('Missing scaffold local trace')
            direct[2] = self.scaffold_pool[sc['local'][rid]]
            context[2] = self.scaffold_pool[sc['window_likelihood_molecules']]
            panels[2] = self.scaffold_pool[sc['conditioning_haplotypes']]
        for data, experts in [(refinement, (3,5)), (spanning, (8,9))]:
            if any(available[e] for e in experts):
                if data is None or 'lineage_molecules' not in data['samples'][sample]:
                    raise ValueError('Available refinement missing actual read inventory')
                values = frozenset(data['samples'][sample]['lineage_molecules'])
                for e in experts:
                    direct[e] = context[e] = values
        joint = self.joint.get(sample)
        if available[7]:
            if joint is None or rid not in joint['known']:
                raise ValueError('Missing joint LD trace')
            direct[7], context[7], panels[7] = local, joint['window'], joint['panel']
        result = np.zeros((10, len(NAMES)), np.float32)
        cohort = window.cohort(sample, record['record_index'], local)
        for e in range(10):
            if not available[e]:
                continue
            if direct[e] is None:
                raise ValueError('Available expert missing direct inventory')
            reused = sum(window.reuse[sample][m] > 1 for m in direct[e]) / len(direct[e]) if direct[e] else 0.
            whole = (e in (8,9) and spanning['samples'][sample].get('lineage_kind') == 'whole_read_uniform_prior')
            result[e, :16] = [1., log_count(direct[e]), float(context[e] is not None), log_count(context[e]),
                reused, float(panels[e] is not None), log_count(panels[e]),
                *(cohort if e == 0 else [0.]*5),
                joint['known'][rid] if e == 7 else 0.,
                float(len(window.trace['catalog_parent_samples']) > 1),
                float(info['library_namespace_complete']), float(whole)]
            for peer in range(10):
                if available[peer]:
                    # Cross-stage molecule namespaces are only compared when
                    # RG/LB identifies them consistently across implementations.
                    d = overlap(direct[e], direct[peer]) if info['library_namespace_complete'] else (0.,0.)
                    p = overlap(panels[e], panels[peer])
                    result[e, 16+4*peer:20+4*peer] = [*d, *p]
        return result


class Exporter:
    def __init__(self, args):
        self.lineage = LineageFeatures(args)
        self.output = Path(args.output).parent / 'trust-trace'
        self.output.mkdir(exist_ok=False)
        self.shards = []

    def write(self, pending, samples, probabilities, spanning):
        columns = {k: [] for k in ['features','available','n_genotypes','probabilities','genotype_features',
                                   'frozen_gp','lineage_features','hmm_features','sample','record_id','input_record_id','start0','alleles','chrom']}
        offset = 0
        for original, refinement, item, entries in pending:
            rid = f'{original.chrom}:{original.pos}:{original.ref}:{",".join(original.alts or [])}'
            for sample, data in zip(samples, entries):
                ng = data['n_genotypes']
                for key in ['features','available','n_genotypes']:
                    columns[key].append(data[key])
                columns['probabilities'].append(data['probabilities'][:, :ng].T)
                columns['genotype_features'].append(data['genotype_features'][:ng])
                columns['frozen_gp'].append(probabilities[offset, :ng].copy())
                columns['lineage_features'].append(self.lineage.row(original, sample, refinement, spanning.get(rid), data['available']))
                window=self.lineage.lookup[sample,rid]
                columns['hmm_features'].append(window.rows[sample,rid]['hmm_uncertainty'])
                columns['sample'].append(sample); columns['input_record_id'].append(rid)
                columns['record_id'].append(f'{item.chrom}:{item.pos}:{item.ref}:{",".join(item.alts or [])}')
                columns['start0'].append(item.start); columns['alleles'].append(json.dumps(list(item.alleles), separators=(',',':')))
                columns['chrom'].append(item.chrom)
                offset += 1
        arrays = {k: np.concatenate(v) if k in ('probabilities','genotype_features','frozen_gp') else np.asarray(v)
                  for k,v in columns.items()}
        arrays['offsets'] = np.r_[0, np.cumsum(arrays['n_genotypes'])]
        path = self.output / f'trace.{len(self.shards):05d}.npz'
        np.savez_compressed(path, **arrays)
        self.shards.append(dict(file=path.name, sha256=sha(path), rows=offset))

    def finish(self):
        from posterior_summary import NAMES as HMM_NAMES
        value = dict(version=VERSION, lineage_feature_names=NAMES, shards=self.shards,
            hmm_feature_names=HMM_NAMES,
            input_sha256=self.lineage.inputs, truth_used=False, weights_changed=False,
            scope='Direct read and conditioning-state overlaps, conservative catalog and selected cohort context; no claim of conditional independence or complete training ancestry')
        (self.output / 'MANIFEST.json').write_text(json.dumps(value, indent=2)+'\n')
