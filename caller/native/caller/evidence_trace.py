"""Read-only provenance instrumentation for the publication experiment.

Identifiers are joins, never numeric predictors. Exact direct read inventories
are distinct from conservative upstream catalog/window dependencies. A shared
input is not asserted to be a correlation coefficient or independent evidence.
"""
import gzip
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

VERSION = 'evidence-trace-v1'


def enabled():
    return os.environ.get('VARDEPTH_TRACE_EVIDENCE', '0') == '1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


@lru_cache(maxsize=262144)
def molecule_token(sample, library, query_name):
    return hashlib.sha256(canonical(['molecule', sample, library, query_name]).encode()).hexdigest()


def molecules(sample, keys):
    return sorted({molecule_token(str(sample), str(key[0]), str(key[1])) for key in keys})


def donors(panel_sha256, chrom, names, indices):
    ids = []
    for value in indices:
        index = int(value)
        if not 0 <= index < 2 * len(names):
            raise ValueError('Conditioning haplotype outside named reference')
        ids.append(hashlib.sha256(canonical(['panel_haplotype', panel_sha256, chrom,
                                             str(names[index // 2]), index % 2]).encode()).hexdigest())
    return sorted(set(ids))


class Inventories:
    def __init__(self):
        self.values = {}

    def add(self, tokens):
        values = sorted(set(tokens))
        ident = hashlib.sha256(canonical(values).encode()).hexdigest()
        if ident in self.values and self.values[ident] != values:
            raise ValueError('Inventory digest collision')
        self.values[ident] = values
        return ident


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
        json.dump(value, stream, separators=(',', ':'))
    temporary.replace(path)


def read(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def sidecar(path):
    return Path(str(path) + '.evidence.json.gz')


class BaseTrace:
    def __init__(self, base, args, identities):
        self.base = base
        self.pool = Inventories()
        self.rows = []
        self.inputs = identities
        self.chrom = base['manifest']['chrom']
        panel_sha = identities[args.panel]['sha256'] if args.panel else None
        self.window = {}
        for sample in base['samples']:
            raw = [(library, read.query_name) for library, read in base['raw'][sample]]
            likelihood = set()
            for items in base['observations'][sample][4].values():
                likelihood.update(items)
            for block in base['sequence_data'][sample].values():
                likelihood.update(block['fragment_keys'])
            chosen = base['manifest']['selected_external_haplotypes'][sample]
            self.window[sample] = dict(raw_molecules=self.pool.add(molecules(sample, raw)),
                likelihood_molecules=self.pool.add(molecules(sample, likelihood)),
                conditioning_haplotypes=self.pool.add(donors(panel_sha, self.chrom, base['panel_names'], chosen) if panel_sha else []),
                library_namespace_complete=not any(w.startswith(sample + ': missing library metadata')
                    for w in base['manifest']['metadata_warnings']))
        self.panel_sha256 = panel_sha

    def add(self, record, sample, keys, neural_tokens):
        summaries=self.base['initial'][sample][2].get('hmm_uncertainty')
        if summaries is None:
            raise ValueError('v14 requires an actual block-HMM posterior summary')
        self.rows.append(dict(record_id=record['record_id'], record_index=record['record_index'], sample=sample,
            local_molecules=self.pool.add(molecules(sample, keys)), neural_molecules=self.pool.add(neural_tokens),
            catalog_origin=record['source'],hmm_uncertainty=summaries[record['first_seed']].tolist()))

    def finish(self, output):
        cohort = self.base.get('_lineage_cohort')
        if cohort is not None:
            cohort = dict(samples=cohort['samples'], donor_indices=cohort['donor_indices'].tolist(),
                donor_weights=cohort['donor_weights'].tolist(), flank_indices=cohort['flank_indices'])
        dump(Path(output) / 'evidence-trace.json.gz', dict(version=VERSION, inventories=self.pool.values,
            rows=self.rows, windows=self.window, inputs=self.inputs, panel_sha256=self.panel_sha256, cohort=cohort,
            catalog_parent_samples=self.base['samples'],
            scope='Exact local likelihood/neural selected molecule sets and conditioning panel haplotypes; raw window reads are conservative candidate/selection ancestry; cohort eligibility depends on all supplied people',
            model_training_ancestry_complete=False))


class ScaffoldTrace:
    def __init__(self, args, panel_names, identities):
        self.pool = Inventories(); self.samples = {}; self.args = args
        self.names = panel_names; self.identities = identities

    def add(self, sample, variants, observations, selected, raw):
        all_keys = set()
        local = {}
        for i, (position, ref, alt) in enumerate(variants):
            keys = observations.get(i, {})
            all_keys.update(keys)
            rid = f'{self.args.region.split(":")[0]}:{position+1}:{ref}:{alt}'
            local[rid] = self.pool.add(molecules(sample, keys))
        self.samples[sample] = dict(local=local,
            window_likelihood_molecules=self.pool.add(molecules(sample, all_keys)),
            raw_molecules=self.pool.add(molecules(sample, [(library, read.query_name) for library, read in raw])),
            conditioning_haplotypes=self.pool.add(donors(self.identities[self.args.panel],
                self.args.region.split(':')[0], self.names, selected)))

    def finish(self):
        dump(sidecar(self.args.output), dict(version=VERSION, inventories=self.pool.values, samples=self.samples,
            inputs_sha256=self.identities, scope='Actual scaffold local GL inputs and selected conditioning haplotypes; no posterior donor-state occupancy recorded'))
