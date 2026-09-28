// Stable evaluation of the same conditional copying model.
// Consecutive sites with exactly zero refresh share a donor state. Their
// likelihood is accumulated in log space before the boundary HMM is solved.
// No positive refresh is rounded to zero; all sites and all donors remain.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include "hmm_buckets.hpp"

static int gi(int a,int b){if(a>b)std::swap(a,b);return b*(b+1)/2+a;}

extern "C" int categorical_gibbs(int m,int k,const uint8_t* hap,
 const int32_t* counts,const int64_t* offsets,const double* gl,
 const double* refresh,double mutation,int burn,int retained,int chains,
 uint64_t seed,double* gp){
 if(m<1||k<1||burn<0||retained<1||chains<1||mutation<0||mutation>=1)return 1;
 const size_t total=offsets[m];std::fill(gp,gp+total,0.);
 std::vector<int> begin{0},mono(m),neutral(m);
 for(int t=1;t<m;++t)if(refresh[t]>0.)begin.push_back(t);
 begin.push_back(m);const int blocks=int(begin.size())-1;
 std::vector<double> likelihood(total);
 std::vector<size_t> site_offset(m+1),table_offset(m+1);
 for(int t=0;t<m;++t){
  int n=counts[t];if(n<2||n>24||offsets[t+1]-offsets[t]!=n*(n+1)/2)return 2;
  if(!std::isfinite(refresh[t])||refresh[t]<0.||refresh[t]>1.)return 3;
  double top=*std::max_element(gl+offsets[t],gl+offsets[t+1]);
  for(int64_t j=offsets[t];j<offsets[t+1];++j)likelihood[j]=std::exp(std::max(-690.,gl[j]-top));
  mono[t]=hap[size_t(t)*k];
  for(int a=0;a<k;++a){int d=hap[size_t(t)*k+a];if(d>=n&&d!=255)return 3;if(d!=hap[size_t(t)*k])mono[t]=-1;}
  neutral[t]=1;for(int64_t j=offsets[t]+1;j<offsets[t+1];++j)if(likelihood[j]!=likelihood[offsets[t]])neutral[t]=0;
  site_offset[t+1]=site_offset[t]+n*n;
  table_offset[t+1]=table_offset[t]+n*(n+1);
 }
 std::vector<double> sites(site_offset[m]),table(table_offset[m]),logtable(table_offset[m]);
 for(int t=0;t<m;++t){int n=counts[t];for(int other=0;other<n;++other){
  double mean=0.;double* s=sites.data()+site_offset[t]+other*n;
  for(int a=0;a<n;++a){s[a]=likelihood[offsets[t]+gi(a,other)];mean+=s[a]/n;}
  double* e=table.data()+table_offset[t]+other*(n+1);
  for(int d=0;d<n;++d)e[d]=(1-mutation)*s[d]+mutation*mean;
  e[n]=mean;
  for(int d=0;d<=n;++d)logtable[table_offset[t]+other*(n+1)+d]=std::log(e[d]);
 }}
 std::vector<double> emissions(size_t(blocks)*k),forward(size_t(blocks)*k),back(k),conditional(k),posterior(k);
 std::mt19937_64 rng(seed);std::uniform_real_distribution<double> uniform(0.,1.);
 auto draw=[&](const double* p,int n){double z=0.;for(int j=0;j<n;++j)z+=p[j];double u=uniform(rng)*z;
  for(int j=0;j<n-1;++j){u-=p[j];if(u<=0.)return j;}return n-1;};
 double weights[24];
 for(int chain=0;chain<chains;++chain){
  std::vector<int> allele[2]={std::vector<int>(m),std::vector<int>(m)};
  for(int c=0;c<2;++c)for(int t=0;t<m;++t){int d=hap[size_t(t)*k+(2*chain+c)%k],n=counts[t];
   for(int a=0;a<n;++a)weights[a]=d==255?1./n:mutation/n+(a==d?1-mutation:0.);
   allele[c][t]=draw(weights,n);
  }
  for(int iteration=0;iteration<burn+retained;++iteration)for(int chromosome=0;chromosome<2;++chromosome){
   const auto& other=allele[1-chromosome];
   std::fill(emissions.begin(),emissions.end(),0.);
   for(int b=0;b<blocks;++b){
    double* e=emissions.data()+size_t(b)*k;
    for(int t=begin[b];t<begin[b+1];++t){
     if(mono[t]>=0||neutral[t])continue; // common donor-independent factor cancels
     int n=counts[t];const uint8_t* h=hap+size_t(t)*k;
     const double* values=logtable.data()+table_offset[t]+other[t]*(n+1);
     for(int a=0;a<k;++a)e[a]+=values[h[a]==255?n:h[a]];
    }
    double top=*std::max_element(e,e+k);
    for(int a=0;a<k;++a)e[a]=std::exp(e[a]-top);
    double* f=forward.data()+size_t(b)*k;
    double r=refresh[begin[b]],z=0.;
    if(b){const double* previous=f-k;for(int a=0;a<k;++a){f[a]=((1-r)*previous[a]+r/k)*e[a];z+=f[a];}}
    else for(int a=0;a<k;++a){f[a]=e[a]/k;z+=f[a];}
    if(!(z>0.)||!std::isfinite(z))return 4;
    for(int a=0;a<k;++a)f[a]/=z;
   }
   if(iteration>=burn){
    std::fill(back.begin(),back.end(),1.);
    for(int b=blocks-1;b>=0;--b){
     const double* f=forward.data()+size_t(b)*k;double z=0.;
     for(int a=0;a<k;++a){posterior[a]=f[a]*back[a];z+=posterior[a];}
     if(!(z>0.)||!std::isfinite(z))return 5;
     for(int a=0;a<k;++a)posterior[a]/=z;
     for(int t=begin[b+1]-1;t>=begin[b];--t){
      int n=counts[t];const double* s=sites.data()+site_offset[t]+other[t]*n;
      const double* e=table.data()+table_offset[t]+other[t]*(n+1);
      double buckets[256]={};
      if(mono[t]>=0)buckets[mono[t]]=1.;
      else allele_buckets(hap+size_t(t)*k,posterior.data(),k,n,buckets);
      for(int a=0;a<n;++a){
       double q=buckets[255]*(s[a]/n/e[n]);
       for(int d=0;d<n;++d)q+=buckets[d]*(s[a]*(mutation/n+(a==d?1-mutation:0.))/e[d]);
       gp[offsets[t]+gi(a,other[t])]+=q;
      }
     }
     if(b){
      const double* e=emissions.data()+size_t(b)*k;double mean=0.,top=0.;double r=refresh[begin[b]];
      for(int a=0;a<k;++a){conditional[a]=back[a]*e[a];mean+=conditional[a]/k;}
      for(int a=0;a<k;++a){back[a]=(1-r)*conditional[a]+r*mean;top=std::max(top,back[a]);}
      if(!(top>0.)||!std::isfinite(top))return 5;
      for(int a=0;a<k;++a)back[a]/=top;
     }
    }
   }
   int block=blocks-1,donor=draw(forward.data()+size_t(block)*k,k);
   for(int t=m-1;t>=0;--t){
    int n=counts[t],d=hap[size_t(t)*k+donor];const double* s=sites.data()+site_offset[t]+other[t]*n;
    for(int a=0;a<n;++a)weights[a]=s[a]*(d==255?1./n:mutation/n+(a==d?1-mutation:0.));
    allele[chromosome][t]=draw(weights,n);
    if(t){
     double u=uniform(rng); // every original per-site transition draw is consumed
     if(t==begin[block]){
      const double* previous=forward.data()+size_t(block-1)*k;
      double stay=(1-refresh[t])*previous[donor],change=refresh[t]/k;
      if(u*(stay+change)>=stay)donor=draw(previous,k);
      --block;
     }
     // An exactly zero-refresh edge cannot change donor, even if a
     // prefix's binary64 forward probability would have underflowed.
    }
   }
  }
 }
 double expected=double(chains)*retained*2;
 for(int t=0;t<m;++t){double z=0.;for(int64_t j=offsets[t];j<offsets[t+1];++j)z+=gp[j];
  if(!(z>0.)||!std::isfinite(z)||std::abs(z-expected)>expected*1e-5)return 6;
  for(int64_t j=offsets[t];j<offsets[t+1];++j)gp[j]/=z;
 }
 return 0;
}
