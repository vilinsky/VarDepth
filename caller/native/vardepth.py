"""Entry point for the frozen research bundle; forward runtime inputs safely."""
import hashlib,importlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def verify():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for name,expected in manifest['files_sha256'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid manifest path')
        digest=hashlib.sha256()
        with (ROOT/relative).open('rb') as f:
            for block in iter(lambda:f.read(1<<20),b''):digest.update(block)
        actual=digest.hexdigest()
        if actual!=expected:raise ValueError('Bundle file changed: '+name)
    return manifest
def main():
    args=sys.argv[1:]
    if args and args[0]=='phase':
        if not any(x in args for x in ['-h','--help']):verify()
        raise SystemExit(subprocess.call([sys.executable,str(ROOT/'caller/phase_cli.py'),*args[1:]]))
    if args==['--version']:
        print(json.loads((ROOT/'MANIFEST.json').read_text())['version']);return
    if args==['--self-check']:
        manifest=verify()
        missing=[name for name in ['samtools','bcftools','c++','pkg-config'] if shutil.which(name) is None]
        python_versions={};missing_python={}
        for name in ['numpy','pysam','torch']:
            try:python_versions[name]=importlib.import_module(name).__version__
            except Exception as error:missing_python[name]=type(error).__name__+': '+str(error)
        htslib_available=('pkg-config' not in missing and subprocess.run(['pkg-config','--exists','htslib'],capture_output=True).returncode==0)
        print(json.dumps(dict(manifest_verified=True,files=len(manifest['files_sha256']),missing_external_executables=missing,
            python_versions=python_versions,python_import_errors=missing_python,htslib_development_available=htslib_available,
            tested_runtime=manifest['runtime'],scope=manifest['scope']),indent=2))
        if missing or missing_python or not htslib_available:raise SystemExit(2)
        return
    forbidden=['--checkpoint','--trust-checkpoint','--config']
    if any(x==key or x.startswith(key+'=') for x in args for key in forbidden):
        raise ValueError('This entry point uses bundled weights/config; use caller/genome.py explicitly for a different research model')
    if not any(x in args for x in ['-h','--help']):verify()
    command=[sys.executable,str(ROOT/'caller/genome.py'),'--checkpoint',str(ROOT/'model/read-model.pt'),
             '--trust-checkpoint',str(ROOT/'model/fusion-model.pt'),'--config',str(ROOT/'selected_config.json'),*args]
    environment=dict(os.environ)
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:environment.setdefault(name,'1')
    raise SystemExit(subprocess.call(command,env=environment))
if __name__=='__main__':main()
