"""Fast biallelic SNV features; general indel/multiallelic logic is unchanged."""
import ctypes,math
from functools import lru_cache
from pathlib import Path
import numpy as np
from native_build import build_library
@lru_cache(maxsize=1)
def library():
 lib=ctypes.CDLL(str(build_library(Path(__file__).with_name('trust_snv.cpp'))))
 d=ctypes.POINTER(ctypes.c_double);f=ctypes.POINTER(ctypes.c_float);b=ctypes.POINTER(ctypes.c_uint8)
 lib.snv_bundle.argtypes=[d,ctypes.c_int,*([ctypes.c_double]*5),ctypes.c_int,f,b,f,f];lib.snv_bundle.restype=ctypes.c_int
 return lib

def bundle(record,sample,scaffold,joint_ld,max_gp):
 call=record.samples[sample];values=[call['GP'],call.get('BGP',call['GP']),scaffold,joint_ld,call.get('PL')]
 if any(v is not None and len(v)!=3 for v in values):return None
 source=np.zeros((5,3),np.float64);flags=0
 for i,v in enumerate(values):
  if v is not None:source[i]=v;flags|=1<<i
 af=record.info.get('PANEL_AF',(None,))[0];af=max(.001,float(af)) if af is not None else .001
 ad=call.get('AD',(0,0));dp=float(call.get('DP',0) or 0);cs=float(call.get('CS',0) or 0)
 if ad is None or len(ad)!=2:return None
 ps=np.empty((10,3),np.float32);available=np.empty(10,bool);features=np.empty((10,42),np.float32);genotype=np.empty((3,20),np.float32)
 ptr=lambda x,t:x.ctypes.data_as(ctypes.POINTER(t))
 code=library().snv_bundle(ptr(source,ctypes.c_double),flags,dp,cs,af,float(ad[0]),float(ad[1]),record.info.get('SOURCE')=='READS',ptr(ps,ctypes.c_float),ptr(available,ctypes.c_uint8),ptr(features,ctypes.c_float),ptr(genotype,ctypes.c_float))
 if code:raise ValueError('Invalid biallelic expert features: '+str(code))
 # Keep NumPy's float32 logarithm and summation, including zero-mass experts.
 entropy=-(ps*np.log(np.maximum(ps,1e-30))).sum(axis=1)
 features[:,15]=entropy.astype(np.float64)/math.log(3)
 if max_gp==3:wide=ps;gf=genotype
 else:
  wide=np.zeros((10,max_gp),np.float32);wide[:,:3]=ps
  gf=np.zeros((max_gp,20),np.float32);gf[:3]=genotype
 return dict(features=features,probabilities=wide,available=available,n_genotypes=3,indel=False,
             alleles=list(record.alleles),start0=record.start,genotype_features=gf)
