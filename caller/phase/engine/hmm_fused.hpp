// Fused FP64 backward step. Reads and writes each donor once; reductions
// retain the established eight-lane buckets and two-stream conditional mean.
#pragma once
#if defined(__AVX512F__)
#include <immintrin.h>
static bool fused_backward(const uint8_t* h,const double* f,double* conditional,
 int k,int n,double decay,double mean,const double* table,double inverse_scale,
 double* buckets,double& next_mean){
 if(n>4||k<64)return false;
 __m512d sums[5]={};__m512d means[2]={};
 bool exact_reciprocal=(k&(k-1))==0;
 __m512d rd=_mm512_set1_pd(decay),base=_mm512_set1_pd((1-decay)*mean),inv=_mm512_set1_pd(inverse_scale),div=_mm512_set1_pd(double(k)),inv_k=_mm512_set1_pd(1./k);
 int a=0;
 for(;a+8<=k;a+=8){
  __m512i codes=_mm512_cvtepu8_epi64(_mm_loadl_epi64((const __m128i*)(h+a)));
  __m512d back=_mm512_add_pd(_mm512_mul_pd(rd,_mm512_loadu_pd(conditional+a)),base);
  __m512d mass=_mm512_mul_pd(_mm512_loadu_pd(f+a),back),em=_mm512_set1_pd(table[255]);
  for(int j=0;j<n;++j){
   __mmask8 selected=_mm512_cmpeq_epi64_mask(codes,_mm512_set1_epi64(j));
   em=_mm512_mask_mov_pd(em,selected,_mm512_set1_pd(table[j]));
   sums[j]=_mm512_add_pd(sums[j],_mm512_maskz_mov_pd(selected,mass));
  }
  sums[4]=_mm512_add_pd(sums[4],_mm512_maskz_mov_pd(_mm512_cmpeq_epi64_mask(codes,_mm512_set1_epi64(255)),mass));
  __m512d updated=_mm512_mul_pd(_mm512_mul_pd(back,em),inv);
  _mm512_storeu_pd(conditional+a,updated);
  // Reciprocal powers of two are exact in binary64: this preserves each
  // divided value (including rounding), without the SIMD divider bottleneck.
  __m512d normalized=exact_reciprocal?_mm512_mul_pd(updated,inv_k):_mm512_div_pd(updated,div);
  int stream=(a/8)%2;means[stream]=_mm512_add_pd(means[stream],normalized);
 }
 double values[16];
 for(int j=0;j<n;++j){_mm512_storeu_pd(values,sums[j]);for(int lane=0;lane<8;++lane)buckets[j]+=values[lane];}
 _mm512_storeu_pd(values,sums[4]);for(int lane=0;lane<8;++lane)buckets[255]+=values[lane];
 _mm512_storeu_pd(values,means[0]);_mm512_storeu_pd(values+8,means[1]);next_mean=0.;for(double v:values)next_mean+=v;
 for(;a<k;++a){double back=decay*conditional[a]+(1-decay)*mean;double mass=f[a]*back;
  conditional[a]=(back*table[h[a]])*inverse_scale;buckets[h[a]]+=mass;next_mean+=conditional[a]/k;}
 return true;
}
#else
static bool fused_backward(const uint8_t*,const double*,double*,int,int,double,double,const double*,double,double*,double&){return false;}
#endif
