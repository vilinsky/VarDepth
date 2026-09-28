// Controlled FP64 summation of finite nonnegative HMM masses. This changes
// reduction order only; no fast-math, reassociated recurrences or reduced states.
#pragma once
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
static double hmm_sum(const double* x,int n,double divisor=1.){
 double result=0;int i=0;
#if defined(__AVX512F__)
 if(n>=64){
  bool exact_reciprocal=divisor==1.||(divisor==double(n)&&(n&(n-1))==0);
  __m512d s0=_mm512_setzero_pd(),s1=s0,d=_mm512_set1_pd(divisor),inverse=_mm512_set1_pd(1./divisor);
  for(;i+16<=n;i+=16){
   __m512d a=_mm512_loadu_pd(x+i),b=_mm512_loadu_pd(x+i+8);
   s0=_mm512_add_pd(s0,exact_reciprocal?_mm512_mul_pd(a,inverse):_mm512_div_pd(a,d));
   s1=_mm512_add_pd(s1,exact_reciprocal?_mm512_mul_pd(b,inverse):_mm512_div_pd(b,d));
  }
  double values[16];_mm512_storeu_pd(values,s0);_mm512_storeu_pd(values+8,s1);
  for(double v:values)result+=v;
 }
#endif
 for(;i<n;++i)result+=x[i]/divisor;return result;
}
