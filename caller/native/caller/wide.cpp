// Independently implemented mathematical kernels. No competitor source copied.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>
using V=std::vector<double>;
static constexpr double tiny=1e-300;
extern "C" double read_haplotype(const char* read,const unsigned char* qual,int n,const char* hap,int m,double gap,double ext){
 if(n<=0||m<=0)return -INFINITY;
 V pm(m+1,1.0/(m+1)), pi(m+1,0),pd(m+1,0),cm(m+1),ci(m+1),cd(m+1);
 double logs=0;
 for(int i=1;i<=n;++i){
  std::fill(cm.begin(),cm.end(),0);std::fill(ci.begin(),ci.end(),0);std::fill(cd.begin(),cd.end(),0);
  ci[0]=.25*(gap*pm[0]+ext*pi[0]);
  double e=std::pow(10.,-std::min(60,int(qual[i-1]))/10.);
  for(int j=1;j<=m;++j){
   double emit=read[i-1]==hap[j-1]?1-e:e/3;
   cm[j]=emit*((1-2*gap)*pm[j-1]+(1-ext)*(pi[j-1]+pd[j-1]));
   ci[j]=.25*(gap*pm[j]+ext*pi[j]);
   cd[j]=gap*cm[j-1]+ext*cd[j-1];
  }
  double scale=0;for(int j=0;j<=m;++j)scale=std::max(scale,cm[j]+ci[j]+cd[j]);
  if(scale<=0)return -INFINITY;
  if(scale<0x1p-450){
   for(int j=0;j<=m;++j){cm[j]*=0x1p512;ci[j]*=0x1p512;cd[j]*=0x1p512;}
   logs-=512*std::log(2.0);
  }
  pm.swap(cm);pi.swap(ci);pd.swap(cd);
 }
 double result=0;for(int j=0;j<=m;++j)result+=pm[j]+pi[j];
 return std::log(std::max(tiny,result))+logs;
}
// Row-stochastic independent haploid refresh T=(1-r)I+r/K. For diploid state
// matrix P this applies T' P T in O(K^2), not O(K^4).
static V transition(const V& p,int k,double r){
 V rows(k),cols(k),out(k*k);double total=0;
 for(int a=0;a<k;++a)for(int b=0;b<k;++b){double v=p[a*k+b];rows[a]+=v;cols[b]+=v;total+=v;}
 double q=1-r, u=r/k;
 for(int a=0;a<k;++a)for(int b=0;b<k;++b)out[a*k+b]=q*q*p[a*k+b]+q*u*(rows[a]+cols[b])+u*u*total;
 return out;
}
extern "C" void diploid_transition(const double* in,int k,double r,double* out){V x(in,in+k*k);V y=transition(x,k,r);std::copy(y.begin(),y.end(),out);}
// hap probabilities are [marker][donor]; mutation allows alleles missing from
// every reference haplotype. GL is read-only, natural logs [marker][genotype].
// Joint state/genotype inference: emissions marginalize latent genotype, then
// forward/backward integrates paths. Target likelihood is included exactly once.
extern "C" double copying_hmm(int m,int k,const double* hap,const double* gl,const double* mu,const double* recomb,double* gp,double* phase){
 if(m<=0||k<=0)return NAN;
 const int kk=k*k;V forward(size_t(m)*kk), emission(size_t(m)*kk), scal(m), back(kk,1),prior(kk,1./kk);
 double loglik=0;V vit(kk,-std::log(double(kk)));std::vector<int> trace(size_t(m)*kk),path(m);
 for(int t=0;t<m;++t){
  double maxl=std::max({gl[3*t],gl[3*t+1],gl[3*t+2]});
  V l={std::exp(gl[3*t]-maxl),std::exp(gl[3*t+1]-maxl),std::exp(gl[3*t+2]-maxl)};
  if(t)prior=transition(prior,k,recomb[t]);
  double z=0;
  for(int a=0;a<k;++a)for(int b=0;b<k;++b){
   int s=a*k+b;double x=mu[t]+(1-2*mu[t])*hap[t*k+a],y=mu[t]+(1-2*mu[t])*hap[t*k+b];
   double e=(1-x)*(1-y)*l[0]+(x*(1-y)+(1-x)*y)*l[1]+x*y*l[2];
   emission[size_t(t)*kk+s]=std::max(tiny,e);prior[s]*=std::max(tiny,e);z+=prior[s];
  }
  if(!(z>0))return NAN;scal[t]=z;loglik+=std::log(z)+maxl;
  for(int s=0;s<kk;++s)forward[size_t(t)*kk+s]=prior[s]/z;
  for(int s=0;s<kk;++s)prior[s]/=z;
  if(t){
   double u=std::max(1e-12,recomb[t]/k), diag=std::log(1-recomb[t]+u),off=std::log(u);
   V mid(kk);std::vector<int> origin(kk);
   for(int b=0;b<k;++b){
    int top=0,second=k>1?1:0;if(vit[second*k+b]>vit[top*k+b])std::swap(top,second);
    for(int a=2;a<k;++a){if(vit[a*k+b]>vit[top*k+b]){second=top;top=a;}else if(vit[a*k+b]>vit[second*k+b])second=a;}
    for(int c=0;c<k;++c){int other=top==c?second:top;double stay=vit[c*k+b]+diag,change=vit[other*k+b]+off;
     int a=stay>=change?c:other;mid[c*k+b]=std::max(stay,change);origin[c*k+b]=a;}
   }
   for(int c=0;c<k;++c){
    int top=0,second=k>1?1:0;if(mid[c*k+second]>mid[c*k+top])std::swap(top,second);
    for(int b=2;b<k;++b){if(mid[c*k+b]>mid[c*k+top]){second=top;top=b;}else if(mid[c*k+b]>mid[c*k+second])second=b;}
    for(int d=0;d<k;++d){int other=top==d?second:top;double stay=mid[c*k+d]+diag,change=mid[c*k+other]+off;
     int b=stay>=change?d:other;vit[c*k+d]=std::max(stay,change)+std::log(emission[size_t(t)*kk+c*k+d]);
     trace[size_t(t)*kk+c*k+d]=origin[c*k+b]*k+b;}
   }
  }else for(int s=0;s<kk;++s)vit[s]+=std::log(emission[s]);
 }
 path[m-1]=int(std::max_element(vit.begin(),vit.end())-vit.begin());
 for(int t=m-1;t>0;--t)path[t-1]=trace[size_t(t)*kk+path[t]];
 for(int t=m-1;t>=0;--t){
  double maxl=std::max({gl[3*t],gl[3*t+1],gl[3*t+2]});
  double l[3]={std::exp(gl[3*t]-maxl),std::exp(gl[3*t+1]-maxl),std::exp(gl[3*t+2]-maxl)};
  double sumg[3]={0,0,0}, norm=0;
  for(int a=0;a<k;++a)for(int b=0;b<k;++b){
   int s=a*k+b;double w=forward[size_t(t)*kk+s]*back[s]/emission[size_t(t)*kk+s];
   double x=mu[t]+(1-2*mu[t])*hap[t*k+a],y=mu[t]+(1-2*mu[t])*hap[t*k+b];
   sumg[0]+=w*(1-x)*(1-y)*l[0];sumg[1]+=w*(x*(1-y)+(1-x)*y)*l[1];sumg[2]+=w*x*y*l[2];
  }
  norm=sumg[0]+sumg[1]+sumg[2];for(int g=0;g<3;++g)gp[t*3+g]=sumg[g]/norm;
  // Coherent Viterbi path-conditioned allele marginals, not phase confidence.
  int best=path[t];
  double x=mu[t]+(1-2*mu[t])*hap[t*k+best/k],y=mu[t]+(1-2*mu[t])*hap[t*k+best%k];
  double e=(1-x)*(1-y)*l[0]+(x*(1-y)+(1-x)*y)*l[1]+x*y*l[2];
  phase[t*2]=(x*(1-y)*l[1]+x*y*l[2])/std::max(tiny,e);
  phase[t*2+1]=((1-x)*y*l[1]+x*y*l[2])/std::max(tiny,e);
  if(t){for(int s=0;s<kk;++s)back[s]*=emission[size_t(t)*kk+s]/scal[t];back=transition(back,k,recomb[t]);}
 }
 return loglik;
}
// General block HMM. eg[marker,state,genotype] is the unnormalized joint
// read/block likelihood and genotype mass. Marker-to-block mapping is explicit.
extern "C" double block_hmm(int blocks,int markers,int k,const double* emissions,const double* eg,const int* block_of,const double* recomb,double* gp,double* state_posterior){
 int kk=k*k;V fw(size_t(blocks)*kk),scales(blocks),prior(kk,1./kk),back(kk,1);double loglik=0;
 for(int t=0;t<blocks;++t){
  if(t)prior=transition(prior,k,recomb[t]);double z=0;
  for(int s=0;s<kk;++s){prior[s]*=std::max(tiny,emissions[size_t(t)*kk+s]);z+=prior[s];}
  if(z<=0)return NAN;scales[t]=z;loglik+=std::log(z);
  for(int s=0;s<kk;++s){prior[s]/=z;fw[size_t(t)*kk+s]=prior[s];}
 }
 for(int t=blocks-1;t>=0;--t){
  double z=0;for(int s=0;s<kk;++s){double q=fw[size_t(t)*kk+s]*back[s];state_posterior[size_t(t)*kk+s]=q;z+=q;}
  for(int s=0;s<kk;++s)state_posterior[size_t(t)*kk+s]/=z;
  if(t){for(int s=0;s<kk;++s)back[s]*=emissions[size_t(t)*kk+s]/scales[t];back=transition(back,k,recomb[t]);}
 }
 for(int j=0;j<markers;++j){int t=block_of[j];double total=0;
  for(int g=0;g<3;++g){double q=0;for(int s=0;s<kk;++s)q+=state_posterior[size_t(t)*kk+s]*eg[(size_t(j)*kk+s)*3+g]/std::max(tiny,emissions[size_t(t)*kk+s]);gp[j*3+g]=q;total+=q;}
  for(int g=0;g<3;++g)gp[j*3+g]/=total;
 }
 return loglik;
}

// Exact two-lane evaluation of the existing PairHMM recurrence. No pruning,
// banding, precision reduction, reassociation, or change to model probabilities.
// SSE2 is baseline x86_64; other architectures retain the scalar implementation.
#if defined(__x86_64__)
#include <emmintrin.h>
static void read_haplotype_pair(const char* read,const unsigned char* qual,int n,
                               const char* h0,int m0,const char* h1,int m1,
                               double gap,double ext,double* out){
 if(n<=0||m0<=0||m1<=0){
  out[0]=read_haplotype(read,qual,n,h0,m0,gap,ext);
  out[1]=read_haplotype(read,qual,n,h1,m1,gap,ext);return;
 }
 const int m=std::max(m0,m1);
 using Q=__m128d;
 std::vector<Q> pm(m+1),pi(m+1),pd(m+1),cm(m+1),ci(m+1),cd(m+1);
 const Q zero=_mm_setzero_pd(),quarter=_mm_set1_pd(.25),g=_mm_set1_pd(gap),x=_mm_set1_pd(ext),
         match=_mm_set1_pd(1-2*gap),exit_gap=_mm_set1_pd(1-ext);
 for(int j=0;j<=m;++j){
  pm[j]=_mm_set_pd(j<=m1?1.0/(m1+1):0.,j<=m0?1.0/(m0+1):0.);
  pi[j]=pd[j]=zero;
 }
 double logs0=0.,logs1=0.;
 alignas(16) double values[2];
 for(int i=1;i<=n;++i){
  cm[0]=cd[0]=zero;
  ci[0]=_mm_mul_pd(quarter,_mm_add_pd(_mm_mul_pd(g,pm[0]),_mm_mul_pd(x,pi[0])));
  double e=std::pow(10.,-std::min(60,int(qual[i-1]))/10.);
  Q scale=_mm_max_pd(zero,_mm_add_pd(_mm_add_pd(cm[0],ci[0]),cd[0]));
  for(int j=1;j<=m;++j){
   const Q active=_mm_castsi128_pd(_mm_set_epi64x(j<=m1?-1LL:0LL,j<=m0?-1LL:0LL));
   const Q emit=_mm_set_pd(j<=m1?(read[i-1]==h1[j-1]?1-e:e/3):0.,
                          j<=m0?(read[i-1]==h0[j-1]?1-e:e/3):0.);
   cm[j]=_mm_and_pd(active,_mm_mul_pd(emit,_mm_add_pd(_mm_mul_pd(match,pm[j-1]),
                                _mm_mul_pd(exit_gap,_mm_add_pd(pi[j-1],pd[j-1])))));
   ci[j]=_mm_and_pd(active,_mm_mul_pd(quarter,_mm_add_pd(_mm_mul_pd(g,pm[j]),_mm_mul_pd(x,pi[j]))));
   cd[j]=_mm_and_pd(active,_mm_add_pd(_mm_mul_pd(g,cm[j-1]),_mm_mul_pd(x,cd[j-1])));
   scale=_mm_max_pd(scale,_mm_add_pd(_mm_add_pd(cm[j],ci[j]),cd[j]));
  }
  _mm_store_pd(values,scale);
  if(!(values[0]>0)||!(values[1]>0)){
   out[0]=read_haplotype(read,qual,n,h0,m0,gap,ext);
   out[1]=read_haplotype(read,qual,n,h1,m1,gap,ext);return;
  }
  bool low0=values[0]<0x1p-450,low1=values[1]<0x1p-450;
  if(low0||low1){
   Q factor=_mm_set_pd(low1?0x1p512:1.0,low0?0x1p512:1.0);
   for(int j=0;j<=m;++j){cm[j]=_mm_mul_pd(cm[j],factor);ci[j]=_mm_mul_pd(ci[j],factor);cd[j]=_mm_mul_pd(cd[j],factor);}
   if(low0)logs0-=512*std::log(2.0);
   if(low1)logs1-=512*std::log(2.0);
  }
  pm.swap(cm);pi.swap(ci);pd.swap(cd);
 }
 Q sum=zero;for(int j=0;j<=m;++j)sum=_mm_add_pd(sum,_mm_add_pd(pm[j],pi[j]));
 _mm_store_pd(values,sum);
 out[0]=std::log(std::max(tiny,values[0]))+logs0;
 out[1]=std::log(std::max(tiny,values[1]))+logs1;
}
#endif
extern "C" void read_haplotype_batch(const char* read,const unsigned char* qual,int n,
                                      const char* const* hap,const int* lengths,int count,
                                      double gap,double ext,double* out){
 int k=0;
#if defined(__x86_64__)
 for(;k+1<count;k+=2)read_haplotype_pair(read,qual,n,hap[k],lengths[k],hap[k+1],lengths[k+1],gap,ext,out+k);
#endif
 for(;k<count;++k)out[k]=read_haplotype(read,qual,n,hap[k],lengths[k],gap,ext);
}

// Independently written widening of our existing exact recurrence.
#if defined(__x86_64__)
#include <immintrin.h>
#include <new>
struct WideBuffer {
 void* data;
 explicit WideBuffer(size_t bytes):data(::operator new(bytes,std::align_val_t(64))){}
 ~WideBuffer(){::operator delete(data,std::align_val_t(64));}
 WideBuffer(const WideBuffer&)=delete;
 WideBuffer& operator=(const WideBuffer&)=delete;
};

__attribute__((target("avx2")))
static void read_wide_4(const char* read,const unsigned char* qual,int n,
 const char* const* haps,const int* lengths,double gap,double ext,double* out){
 using Q=__m256d;int m=0;
 for(int lane=0;lane<4;++lane){
  if(n<=0||lengths[lane]<=0){for(int k=0;k<4;++k)out[k]=read_haplotype(read,qual,n,haps[k],lengths[k],gap,ext);return;}
  m=std::max(m,lengths[lane]);
 }
 // Function target attributes do not reliably propagate vector alignment to
 // std::allocator. Allocate all six arrays explicitly at 64-byte alignment.
 WideBuffer memory(8*size_t(m+1)*sizeof(Q));
 Q* pm=(Q*)memory.data;Q* pi=pm+m+1;Q* pd=pi+m+1;
 Q* cm=pd+m+1;Q* ci=cm+m+1;Q* cd=ci+m+1;
 Q* codes=cd+m+1;Q* active_data=codes+m+1;
 std::vector<unsigned char> active_masks(m+1);
 const Q zero=_mm256_setzero_pd(),quarter=_mm256_set1_pd(.25),g=_mm256_set1_pd(gap),x=_mm256_set1_pd(ext),
  match=_mm256_set1_pd(1-2*gap),exit_gap=_mm256_set1_pd(1-ext);
 alignas(64) double values[4],emissions[4];double logs[4]={0};
 for(int j=0;j<=m;++j){
  alignas(64) long long hcodes[4],bits[4];unsigned mask=0;
  for(int lane=0;lane<4;++lane){
   hcodes[lane]=(j>0&&j<=lengths[lane])?(unsigned char)haps[lane][j-1]:0;
   bits[lane]=j<=lengths[lane]?-1LL:0LL;
   if(j<=lengths[lane])mask|=1u<<lane;
  }
  codes[j]=_mm256_castsi256_pd(_mm256_load_si256((const __m256i*)hcodes));active_data[j]=_mm256_castsi256_pd(_mm256_load_si256((const __m256i*)bits));active_masks[j]=mask;
  for(int lane=0;lane<4;++lane)values[lane]=j<=lengths[lane]?1.0/(lengths[lane]+1):0.;
  pm[j]=_mm256_load_pd(values);pi[j]=pd[j]=zero;
 }
 for(int i=1;i<=n;++i){
  cm[0]=cd[0]=zero;
  ci[0]=_mm256_mul_pd(quarter,_mm256_add_pd(_mm256_mul_pd(g,pm[0]),_mm256_mul_pd(x,pi[0])));
  double e=std::pow(10.,-std::min(60,int(qual[i-1]))/10.);
  const __m256i readbase=_mm256_set1_epi64x((unsigned char)read[i-1]);
  const Q good=_mm256_set1_pd(1-e),bad=_mm256_set1_pd(e/3);
  Q scale=_mm256_max_pd(zero,_mm256_add_pd(_mm256_add_pd(cm[0],ci[0]),cd[0]));
  for(int j=1;j<=m;++j){
   const Q active=active_data[j];
   const Q emit=_mm256_blendv_pd(bad,good,_mm256_castsi256_pd(_mm256_cmpeq_epi64(readbase,_mm256_castpd_si256(codes[j]))));
   cm[j]=_mm256_and_pd(active,_mm256_mul_pd(emit,_mm256_add_pd(_mm256_mul_pd(match,pm[j-1]),_mm256_mul_pd(exit_gap,_mm256_add_pd(pi[j-1],pd[j-1])))));ci[j]=_mm256_and_pd(active,_mm256_mul_pd(quarter,_mm256_add_pd(_mm256_mul_pd(g,pm[j]),_mm256_mul_pd(x,pi[j]))));cd[j]=_mm256_and_pd(active,_mm256_add_pd(_mm256_mul_pd(g,cm[j-1]),_mm256_mul_pd(x,cd[j-1])));
   scale=_mm256_max_pd(scale,_mm256_add_pd(_mm256_add_pd(cm[j],ci[j]),cd[j]));
  }
  _mm256_store_pd(values,scale);
  for(int lane=0;lane<4;++lane)if(!(values[lane]>0)){
   for(int k=0;k<4;++k)out[k]=read_haplotype(read,qual,n,haps[k],lengths[k],gap,ext);return;
  }
  // Same full double-precision recurrence; rescale only when underflow is
  // approaching, using an exactly representable power of two. Rounding can
  // differ from per-row division, so numerical and public genotype parity are
  // required before promotion. No paths, reads or alleles are discarded.
  bool rescale=false;
  for(int lane=0;lane<4;++lane){
   const bool low=values[lane]<0x1p-450;
   emissions[lane]=low?0x1p512:1.0;
   if(low){logs[lane]-=512*std::log(2.0);rescale=true;}
  }
  if(rescale){
   const Q factor=_mm256_load_pd(emissions);
   for(int j=0;j<=m;++j){cm[j]=_mm256_mul_pd(cm[j],factor);ci[j]=_mm256_mul_pd(ci[j],factor);cd[j]=_mm256_mul_pd(cd[j],factor);}
  }
  std::swap(pm,cm);std::swap(pi,ci);std::swap(pd,cd);
 }
 Q sum=zero;for(int j=0;j<=m;++j)sum=_mm256_add_pd(sum,_mm256_add_pd(pm[j],pi[j]));
 _mm256_store_pd(values,sum);
 for(int lane=0;lane<4;++lane)out[lane]=std::log(std::max(tiny,values[lane]))+logs[lane];
}

__attribute__((target("avx512f")))
static void read_wide_8(const char* read,const unsigned char* qual,int n,
 const char* const* haps,const int* lengths,double gap,double ext,double* out){
 using Q=__m512d;int m=0;
 for(int lane=0;lane<8;++lane){
  if(n<=0||lengths[lane]<=0){for(int k=0;k<8;++k)out[k]=read_haplotype(read,qual,n,haps[k],lengths[k],gap,ext);return;}
  m=std::max(m,lengths[lane]);
 }
 // Function target attributes do not reliably propagate vector alignment to
 // std::allocator. Allocate all six arrays explicitly at 64-byte alignment.
 WideBuffer memory(8*size_t(m+1)*sizeof(Q));
 Q* pm=(Q*)memory.data;Q* pi=pm+m+1;Q* pd=pi+m+1;
 Q* cm=pd+m+1;Q* ci=cm+m+1;Q* cd=ci+m+1;
 Q* codes=cd+m+1;Q* active_data=codes+m+1;
 std::vector<unsigned char> active_masks(m+1);
 const Q zero=_mm512_setzero_pd(),quarter=_mm512_set1_pd(.25),g=_mm512_set1_pd(gap),x=_mm512_set1_pd(ext),
  match=_mm512_set1_pd(1-2*gap),exit_gap=_mm512_set1_pd(1-ext);
 alignas(64) double values[8],emissions[8];double logs[8]={0};
 for(int j=0;j<=m;++j){
  alignas(64) long long hcodes[8],bits[8];unsigned mask=0;
  for(int lane=0;lane<8;++lane){
   hcodes[lane]=(j>0&&j<=lengths[lane])?(unsigned char)haps[lane][j-1]:0;
   bits[lane]=j<=lengths[lane]?-1LL:0LL;
   if(j<=lengths[lane])mask|=1u<<lane;
  }
  codes[j]=_mm512_castsi512_pd(_mm512_load_si512((const void*)hcodes));active_data[j]=_mm512_castsi512_pd(_mm512_load_si512((const void*)bits));active_masks[j]=mask;
  for(int lane=0;lane<8;++lane)values[lane]=j<=lengths[lane]?1.0/(lengths[lane]+1):0.;
  pm[j]=_mm512_load_pd(values);pi[j]=pd[j]=zero;
 }
 for(int i=1;i<=n;++i){
  cm[0]=cd[0]=zero;
  ci[0]=_mm512_mul_pd(quarter,_mm512_add_pd(_mm512_mul_pd(g,pm[0]),_mm512_mul_pd(x,pi[0])));
  double e=std::pow(10.,-std::min(60,int(qual[i-1]))/10.);
  const __m512i readbase=_mm512_set1_epi64((unsigned char)read[i-1]);
  const Q good=_mm512_set1_pd(1-e),bad=_mm512_set1_pd(e/3);
  Q scale=_mm512_max_pd(zero,_mm512_add_pd(_mm512_add_pd(cm[0],ci[0]),cd[0]));
  for(int j=1;j<=m;++j){
   const unsigned active=active_masks[j];
   const Q emit=_mm512_mask_blend_pd(_mm512_cmpeq_epi64_mask(readbase,_mm512_castpd_si512(codes[j])),bad,good);
   cm[j]=_mm512_maskz_mov_pd((__mmask8)active,_mm512_mul_pd(emit,_mm512_add_pd(_mm512_mul_pd(match,pm[j-1]),_mm512_mul_pd(exit_gap,_mm512_add_pd(pi[j-1],pd[j-1])))));ci[j]=_mm512_maskz_mov_pd((__mmask8)active,_mm512_mul_pd(quarter,_mm512_add_pd(_mm512_mul_pd(g,pm[j]),_mm512_mul_pd(x,pi[j]))));cd[j]=_mm512_maskz_mov_pd((__mmask8)active,_mm512_add_pd(_mm512_mul_pd(g,cm[j-1]),_mm512_mul_pd(x,cd[j-1])));
   scale=_mm512_max_pd(scale,_mm512_add_pd(_mm512_add_pd(cm[j],ci[j]),cd[j]));
  }
  _mm512_store_pd(values,scale);
  for(int lane=0;lane<8;++lane)if(!(values[lane]>0)){
   for(int k=0;k<8;++k)out[k]=read_haplotype(read,qual,n,haps[k],lengths[k],gap,ext);return;
  }
  // Same full double-precision recurrence; rescale only when underflow is
  // approaching, using an exactly representable power of two. Rounding can
  // differ from per-row division, so numerical and public genotype parity are
  // required before promotion. No paths, reads or alleles are discarded.
  bool rescale=false;
  for(int lane=0;lane<8;++lane){
   const bool low=values[lane]<0x1p-450;
   emissions[lane]=low?0x1p512:1.0;
   if(low){logs[lane]-=512*std::log(2.0);rescale=true;}
  }
  if(rescale){
   const Q factor=_mm512_load_pd(emissions);
   for(int j=0;j<=m;++j){cm[j]=_mm512_mul_pd(cm[j],factor);ci[j]=_mm512_mul_pd(ci[j],factor);cd[j]=_mm512_mul_pd(cd[j],factor);}
  }
  std::swap(pm,cm);std::swap(pi,ci);std::swap(pd,cd);
 }
 Q sum=zero;for(int j=0;j<=m;++j)sum=_mm512_add_pd(sum,_mm512_add_pd(pm[j],pi[j]));
 _mm512_store_pd(values,sum);
 for(int lane=0;lane<8;++lane)out[lane]=std::log(std::max(tiny,values[lane]))+logs[lane];
}

#endif
extern "C" int wide_capability(){
#if defined(__x86_64__)
 if(__builtin_cpu_supports("avx512f"))return 8;
 if(__builtin_cpu_supports("avx2"))return 4;
#endif
 return 2;
}
extern "C" void read_haplotype_batch_mode(const char* read,const unsigned char* qual,int n,
 const char* const* haps,const int* lengths,int count,double gap,double ext,double* out,int mode){
 int k=0,limit=mode?mode:wide_capability();
#if defined(__x86_64__)
 if(limit>=8&&__builtin_cpu_supports("avx512f"))for(;k+7<count;k+=8)read_wide_8(read,qual,n,haps+k,lengths+k,gap,ext,out+k);
 if(limit>=4&&__builtin_cpu_supports("avx2"))for(;k+3<count;k+=4)read_wide_4(read,qual,n,haps+k,lengths+k,gap,ext,out+k);
#endif
 read_haplotype_batch(read,qual,n,haps+k,lengths+k,count-k,gap,ext,out+k);
}
