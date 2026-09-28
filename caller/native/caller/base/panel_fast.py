"""Fast phased-panel I/O with the reference Python loader's return contract."""
import collections
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading

import numpy as np

HERE = Path(__file__).resolve().parent


def executable():
    build = HERE / 'build'
    build.mkdir(exist_ok=True)
    binary = build / 'panel_io'
    fingerprint = hashlib.sha256((HERE / 'core/panel_io.cpp').read_bytes() + (HERE / 'build_panel.sh').read_bytes()).hexdigest()
    marker = build / 'panel_io.source.sha256'
    with (build / 'panel_io.build.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not binary.exists() or not marker.exists() or marker.read_text().strip() != fingerprint:
            subprocess.run(['bash', str(HERE / 'build_panel.sh')], check=True)
            marker.write_text(fingerprint + '\n')
    return binary


def _load_panel_fast(path, chrom, start, end, fetch, excluded, max_indel):
    if not path:
        return {}, [], {}
    # Imported lazily to avoid an import cycle when caller selects this loader.
    from caller import normalize_variant
    panel = {}
    skips = collections.Counter()
    with tempfile.TemporaryDirectory(prefix='vardepth-panel-') as directory:
        directory = Path(directory)
        exclusions = directory / 'excluded.txt'
        exclusions.write_text(''.join(str(s) + '\n' for s in sorted(excluded)))
        prefix = directory / 'panel'
        subprocess.run([str(executable()), str(path), f'{chrom}:{start+1}-{end}', str(exclusions),
                        str(prefix), str(max_indel + 1), str(start), str(end)], check=True)
        names = prefix.with_suffix('.names').read_text().splitlines()
        stats = json.loads(prefix.with_suffix('.json').read_text())
        raw = np.fromfile(prefix.with_suffix('.gt'), dtype=np.uint8).reshape(stats['rows'], stats['columns'])
        if stats['columns'] != len(names) * 2:
            raise RuntimeError('Panel metadata dimensions disagree')
        called=(raw!=255).sum(axis=1);alternate=(raw==1).sum(axis=1)
        frequencies=np.divide(alternate,called,out=np.zeros(len(raw)),where=called>0)
        with prefix.with_suffix('.sites').open() as stream:
            for index, line in enumerate(stream):
                p, ref, alt = line.rstrip('\n').split('\t')
                p = int(p)
                if fetch(p, p + len(ref)).upper() != ref:
                    skips['ref_mismatch'] += 1
                    continue
                variant = normalize_variant(p, ref, alt, fetch)
                if variant[0] < start:
                    continue
                row = raw[index]
                af = float(frequencies[index])
                # Fully observed phased alleles are exactly binary. Missing
                # entries retain the original floating frequency imputation.
                if called[index]==stats['columns']:
                    haplotypes = row.copy()
                else:
                    haplotypes = row.astype(np.float64)
                    haplotypes[row==255] = af
                panel[variant] = (haplotypes, af)
                skips['unusable_sample_genotypes'] += int((stats['columns']-called[index]) // 2)
        if stats['unsupported']:
            skips['unsupported'] += stats['unsupported']
    return panel, names, {k: v for k, v in skips.items() if v}


_PANEL_CACHE=collections.OrderedDict()
_PANEL_BYTES=0
_PANEL_LIMIT=256*1024*1024
_PANEL_LOCK=threading.RLock()

def _file_stamp(path):
    p=Path(path).resolve()
    if not p.exists():return (str(p),None)
    t=p.stat()
    return (str(p),t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns)

def _panel_copy(value):
    panel,names,skips=value
    return {v:(h.copy(),af) for v,(h,af) in panel.items()},list(names),dict(skips)

def load_panel_fast(path, chrom, start, end, fetch, excluded, max_indel, *, reference_identity=None):
    """Reuse only an identical panel query under immutable BAM-run inputs.

    Normalized coordinates, sample exclusions, allele frequencies and skip
    counts are retained verbatim. No inference or posterior is cached.
    """
    if not path or reference_identity is None:
        return _load_panel_fast(path,chrom,start,end,fetch,excluded,max_indel)
    global _PANEL_BYTES
    files=[path,str(path)+'.csi',str(path)+'.tbi',reference_identity,
           str(reference_identity)+'.fai',str(reference_identity)+'.gzi']
    key=(tuple(_file_stamp(p) for p in files),chrom,start,end,tuple(sorted(excluded)),max_indel)
    with _PANEL_LOCK:
        saved=_PANEL_CACHE.get(key)
        if saved is not None:
            _PANEL_CACHE.move_to_end(key)
            return _panel_copy(saved[0])
    result=_load_panel_fast(path,chrom,start,end,fetch,excluded,max_indel)
    size=sum(h.nbytes+256 for h,af in result[0].values())
    if size<=_PANEL_LIMIT:
        # Cache owns separate arrays; callers may retain or mutate their copy.
        stored=_panel_copy(result)
        with _PANEL_LOCK:
            previous=_PANEL_CACHE.pop(key,None)
            if previous is not None:_PANEL_BYTES-=previous[1]
            _PANEL_CACHE[key]=(stored,size);_PANEL_BYTES+=size
            while _PANEL_BYTES>_PANEL_LIMIT:
                _,(_,removed)=_PANEL_CACHE.popitem(last=False);_PANEL_BYTES-=removed
    return result
