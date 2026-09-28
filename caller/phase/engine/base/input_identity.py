"""Cache full input checksums across chunks with explicit filesystem identity.

Cache entries are local execution metadata, not a security boundary. Inputs must
be immutable during inference; size/mtime/ctime/inode changes invalidate entries.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def _stat(path):
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns, inode=stat.st_ino, device=stat.st_dev)


def file_identity(path, cache_dir=None):
    path = Path(path).resolve()
    before = _stat(path)
    cache_dir = cache_dir or os.environ.get('VARDEPTH_HASH_CACHE_DIR')
    def calculate():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if _stat(path) != before:
            raise RuntimeError('Input changed while hashing: ' + str(path))
        return dict(path=str(path), size=before['size'], sha256=digest.hexdigest())
    if not cache_dir:
        return calculate()
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    entry = directory / (key + '.json')
    with (directory / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if entry.exists():
            try:
                saved = json.loads(entry.read_text())
                if saved.get('identity') == before and re.fullmatch('[0-9a-f]{64}', saved['result']['sha256']) and _stat(path) == before:
                    result = saved['result']
                    if result.get('path') == str(path) and result.get('size') == before['size']:
                        return result
            except (ValueError, KeyError, TypeError):
                pass
        result = calculate()
        fd, temporary = tempfile.mkstemp(dir=directory, prefix=key + '.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(dict(identity=before, result=result), stream, sort_keys=True)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, entry)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return result
