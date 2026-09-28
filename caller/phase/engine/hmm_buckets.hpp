// SIMD is confined to posterior accumulation: sampled donor paths are unchanged.
#include <cstring>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
static void allele_buckets(const uint8_t* h,const double* mass,int k,int n,double* buckets){
#if defined(__AVX512F__)
 if(n<=4){
  __m512d b0=_mm512_setzero_pd(),b1=b0,b2=b0,b3=b0,bu=b0;int a=0;
  for(;a+8<=k;a+=8){
   __m512i code=_mm512_cvtepu8_epi64(_mm_loadl_epi64((const __m128i*)(h+a)));__m512d v=_mm512_loadu_pd(mass+a);
   b0=_mm512_add_pd(b0,_mm512_maskz_mov_pd(_mm512_cmpeq_epi64_mask(code,_mm512_setzero_si512()),v));
   b1=_mm512_add_pd(b1,_mm512_maskz_mov_pd(_mm512_cmpeq_epi64_mask(code,_mm512_set1_epi64(1)),v));
   if(n>2)b2=_mm512_add_pd(b2,_mm512_maskz_mov_pd(_mm512_cmpeq_epi64_mask(code,_mm512_set1_epi64(2)),v));
   if(n>3)b3=_mm512_add_pd(b3,_mm512_maskz_mov_pd(_mm512_cmpeq_epi64_mask(code,_mm512_set1_epi64(3)),v));
   bu=_mm512_add_pd(bu,_mm512_maskz_mov_pd(_mm512_cmpeq_epi64_mask(code,_mm512_set1_epi64(255)),v));
  }
  double values[8];
  _mm512_storeu_pd(values,b0);for(double x:values)buckets[0]+=x;
  _mm512_storeu_pd(values,b1);for(double x:values)buckets[1]+=x;
  _mm512_storeu_pd(values,b2);for(double x:values)buckets[2]+=x;
  _mm512_storeu_pd(values,b3);for(double x:values)buckets[3]+=x;
  _mm512_storeu_pd(values,bu);for(double x:values)buckets[255]+=x;
  for(;a<k;++a)buckets[h[a]]+=mass[a];return;
 }
#endif
 for(int a=0;a<k;++a)buckets[h[a]]+=mass[a];
}
