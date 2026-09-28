"""Build the optional BAM-buffer C++ adapter against the installed pysam ABI."""
import argparse,hashlib,json,subprocess,sys,sysconfig
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);a=p.parse_args();root=a.source.resolve()
    import pysam,Cython
    from Cython.Build import cythonize
    from setuptools import Extension,Distribution
    directory=root/'caller';source=directory/'pysam_observations.pyx'
    ext=Extension('pysam_observations',[str(source)],language='c++',
        include_dirs=[str(directory),*pysam.get_include()],define_macros=pysam.get_defines(),
        extra_compile_args=['-O3','-std=c++17','-ffp-contract=off'])
    extensions=cythonize([ext],compiler_directives=dict(language_level=3,annotation_typing=False,infer_types=False,
        boundscheck=True,wraparound=True,initializedcheck=True,cdivision=False,nonecheck=True,cpow=False),
        build_dir=str(root/'native-build/reads/generated'))
    dist=Distribution(dict(name='vardepth-native-observations',ext_modules=extensions))
    cmd=dist.get_command_obj('build_ext');cmd.build_lib=str(directory);cmd.build_temp=str(root/'native-build/reads/objects')
    dist.run_command('build_ext')
    binary=directory/('pysam_observations'+sysconfig.get_config_var('EXT_SUFFIX'))
    rp=root/'native-python.json';receipt=json.loads(rp.read_text()) if rp.exists() else dict(modules={})
    receipt['modules'][str(source.relative_to(root))]=dict(source_sha256=sha(source),binary=str(binary.relative_to(root)),binary_sha256=sha(binary))
    receipt['bam_adapter']=dict(pysam=pysam.__version__,python=sys.version,cython=Cython.__version__,
        source_sha256=sha(source),header_sha256=sha(directory/'native_reads/cigar_targets.hpp'),
        limitations='Immutable pysam BAM records; original fallback for unusual CIGARs and non-pysam objects')
    rp.write_text(json.dumps(receipt,indent=2)+'\n')
    mp=root/'MANIFEST.json';m=json.loads(mp.read_text())
    for item in [source,directory/'native_reads/cigar_targets.hpp',binary,rp]:m['files_sha256'][str(item.relative_to(root))]=sha(item)
    mp.write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(receipt['bam_adapter']),flush=True)
if __name__=='__main__':main()

