// Optional system vector-math ABI, with a scalar libm fallback. No copied math
// implementation, fast-math flags, or assumptions about a particular libm build.
#pragma once
#include <cmath>
#if defined(__linux__) && defined(__AVX512F__)
#include <dlfcn.h>
#include <immintrin.h>
static void vector_log(double* values,size_t n){
 using Function=__m512d(*)(__m512d);
 static Function logarithm=[](){void* handle=dlopen("libmvec.so.1",RTLD_LAZY|RTLD_LOCAL);return handle?reinterpret_cast<Function>(dlsym(handle,"_ZGVeN8v_log")):nullptr;}();
 size_t i=0;
 if(logarithm)for(;i+8<=n;i+=8)_mm512_storeu_pd(values+i,logarithm(_mm512_loadu_pd(values+i)));
 for(;i<n;++i)values[i]=std::log(values[i]);
}
#else
static void vector_log(double* values,size_t n){for(size_t i=0;i<n;++i)values[i]=std::log(values[i]);}
#endif
