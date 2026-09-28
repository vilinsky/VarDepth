"""Process-safe compilation of the independently written v0.3 kernels."""
import fcntl,hashlib,os,pathlib,subprocess,sys
from functools import lru_cache
import platform,shutil

@lru_cache(maxsize=1)
def build_identity():
    compiler=pathlib.Path(shutil.which('c++')).resolve();stat=compiler.stat()
    identity=[str(compiler),str(stat.st_size),str(stat.st_mtime_ns),platform.machine()]
    info=pathlib.Path('/proc/cpuinfo')
    if info.exists():
        selected={}
        for line in info.read_text().splitlines():
            if ':' in line:
                key,value=map(str.strip,line.split(':',1))
                if key in ('model name','flags','Features') and key not in selected:selected[key]=value
        identity.extend(k+'='+v for k,v in sorted(selected.items()))
    return '\n'.join(identity).encode()

def build_library(source):
    source=pathlib.Path(source);directory=source.parent/'build';directory.mkdir(exist_ok=True)
    suffix='dylib' if sys.platform=='darwin' else 'so'
    binary=directory/f'lib{source.stem}.{suffix}';marker=directory/f'{source.stem}.sha256'
    extra=['-ffp-contract=off'] if source.stem in ('trust_snv','mixture_two','selection_sum','batch_hmm','trie_vector','trie_hmm','sparse_pairhmm','categorical','categorical_compressed','categorical_neutral','scaffold','scaffold_compressed','scaffold_neutral') else []
    if source.stem in ('categorical','scaffold') and sys.platform=='linux':extra+=['-march=native','-fopenmp-simd','-mprefer-vector-width=512']
    if source.stem in ('categorical_compressed','categorical_neutral','scaffold_compressed','scaffold_neutral') and sys.platform=='linux':extra+=['-march=native']
    if source.stem=='selection_sum' and sys.platform=='linux':extra+=['-march=native']
    if source.stem=='mixture_general':extra+=['-ffp-contract=off']+(['-march=native','-ldl'] if sys.platform=='linux' else [])
    headers=b''.join(p.read_bytes() for p in sorted(source.parent.glob('*.hpp')))
    fingerprint=hashlib.sha256(source.read_bytes()+headers+str(extra).encode()+build_identity()).hexdigest()
    with (directory/f'{source.stem}.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if not binary.exists() or not marker.exists() or marker.read_text()!=fingerprint:
            tmp=binary.with_name(binary.name+f'.{os.getpid()}.tmp')
            try:
                subprocess.run(['c++','-O3','-std=c++17',*extra,'-dynamiclib' if sys.platform=='darwin' else '-shared','-fPIC',str(source),'-o',str(tmp)],check=True)
                tmp.replace(binary);marker.write_text(fingerprint)
            finally:tmp.unlink(missing_ok=True)
    return binary
