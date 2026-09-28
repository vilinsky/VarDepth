"""Optional native tree traversal, with unchanged NumPy sigmoid/calibration."""
import ctypes
import os
from pathlib import Path
import numpy as np
from features43 import NAMES


class NativeScorer:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.function = self.library.logits44
        arr = np.ctypeslib.ndpointer
        f64 = arr(dtype=np.float64, flags='C_CONTIGUOUS')
        i32 = arr(dtype=np.int32, flags='C_CONTIGUOUS')
        self.function.argtypes = [f64, ctypes.c_int64, ctypes.c_int64,
                                  i32, ctypes.c_int64, i32, i32, i32,
                                  i32, f64, f64, ctypes.c_double, f64]
        self.function.restype = None
        self.cache = {}

    def compile_model(self, model):
        roots=[];nodes=[]
        for tree in model['trees']:
            offset=len(nodes); roots.append(offset)
            for n in tree:
                nodes.append((-1 if n['leaf'] else n['feature'],
                              offset+n['left'], offset+n['right'],
                              int(n['missing_left']), n['threshold'], n['value']))
        columns=list(zip(*nodes))
        return (model, np.asarray(roots,np.int32),
                *[np.asarray(c,np.int32 if i<4 else np.float64) for i,c in enumerate(columns)])

    def __call__(self, model, x):
        x=np.ascontiguousarray(x,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=len(NAMES):raise ValueError('Feature schema mismatch')
        key=id(model)
        if key not in self.cache:self.cache[key]=self.compile_model(model)
        entry=self.cache[key]
        assert entry[0] is model
        roots,feature,left,right,missing,threshold,value=entry[1:]
        out=np.empty(len(x),dtype=np.float64)
        self.function(x,len(x),x.shape[1],roots,len(roots),feature,left,right,
                      missing,threshold,value,model['baseline'],out)
        return 1/(1+np.exp(-np.clip(out,-40,40)))


def activate(library):
    """Explicit opt-in; fixed model dictionaries must not mutate after loading."""
    import policy44
    scorer=NativeScorer(library)
    policy44.infer=scorer
    return scorer


def main():
    import argparse
    from apply44 import apply
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['source','output','model','library']:parser.add_argument('--'+key,required=True)
    a=parser.parse_args();activate(a.library)
    import json
    print(json.dumps(apply(a.source,a.output,a.model)))


if __name__=='__main__':main()
