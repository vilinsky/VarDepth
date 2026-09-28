"""Compile own Python hot loops with conservative Cython semantics.

No typed numeric inference, annotation coercion, fast-math or disabled safety
checks. The resulting extension modules remain subject to VCF parity gates.
"""
import argparse,hashlib,json,platform,subprocess,sys,sysconfig,time
from pathlib import Path

GROUPS={
 'caller/base':['caller','local_alleles','artifact','alignment_queries','context_blocks',
                'likelihood_batch','row_patterns','state_selection','read_index','genetic_map'],
 'caller':['features','base_adapter','projection','cpu_batch','trie_vector','partial_anchor',
           'indel_refine','whole_read','graph_assemble','graph_extend','baseline_proposals',
           'proposals','trust_cpu','trust','joint_ld'],
}

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);a=p.parse_args();root=a.source.resolve()
    import Cython
    from Cython.Build import cythonize
    from setuptools import Extension,Distribution
    started=time.monotonic();records={}
    directives=dict(language_level=3,infer_types=False,annotation_typing=False,
                    binding=True,boundscheck=True,wraparound=True,initializedcheck=True,
                    nonecheck=True,cdivision=False,cpow=False)
    flags=['-O3','-ffp-contract=off']
    for group,names in GROUPS.items():
        directory=root/group;build=root/'native-build'/group.replace('/','_')
        extensions=[Extension(name,[str(directory/(name+'.py'))],extra_compile_args=flags) for name in names]
        extensions=cythonize(extensions,compiler_directives=directives,nthreads=4,
                             build_dir=str(build/'generated'),quiet=False)
        dist=Distribution(dict(name='vardepth-native-loops',ext_modules=extensions))
        cmd=dist.get_command_obj('build_ext');cmd.build_lib=str(directory);cmd.build_temp=str(build/'objects');cmd.parallel=8
        dist.run_command('build_ext')
        for name in names:
            relative=str(Path(group)/(name+'.py'))
            binary=directory/(name+sysconfig.get_config_var('EXT_SUFFIX'))
            if not binary.is_file():raise FileNotFoundError(binary)
            records[relative]=dict(source_sha256=sha(root/relative),binary=str(binary.relative_to(root)),binary_sha256=sha(binary))
    receipt=dict(cython_version=Cython.__version__,python=sys.version,soabi=sysconfig.get_config_var('SOABI'),
        platform=platform.platform(),machine=platform.machine(),compiler=subprocess.check_output(['cc','--version'],text=True).splitlines()[0],
        compiler_directives=directives,extra_compile_args=flags,modules=records,build_seconds=time.monotonic()-started,
        scientific_semantics='Experimental compilation only; no relaxed FP; full output equivalence required')
    path=root/'native-python.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for relative,value in records.items():
        manifest['files_sha256'][relative]=value['source_sha256']
        manifest['files_sha256'][value['binary']]=value['binary_sha256']
    manifest['files_sha256']['native-python.json']=sha(path)
    manifest['native_python_build']=receipt
    (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(modules=len(records),build_seconds=receipt['build_seconds'],cython=Cython.__version__)),flush=True)

if __name__=='__main__':main()
