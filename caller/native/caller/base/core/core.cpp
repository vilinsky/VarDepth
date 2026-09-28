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
  for(int j=0;j<=m;++j){cm[j]/=scale;ci[j]/=scale;cd[j]/=scale;}
  logs+=std::log(scale);pm.swap(cm);pi.swap(ci);pd.swap(cd);
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
  for(int j=0;j<=m;++j){cm[j]=_mm_div_pd(cm[j],scale);ci[j]=_mm_div_pd(ci[j],scale);cd[j]=_mm_div_pd(cd[j],scale);}
  logs0+=std::log(values[0]);logs1+=std::log(values[1]);pm.swap(cm);pi.swap(ci);pd.swap(cd);
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
