// Exact elimination of donor-invariant SNP emissions; original Gibbs RNG order.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include "hmm_cache.hpp"
#include "hmm_sum.hpp"
#include "hmm_buckets.hpp"
#include "hmm_fused.hpp"
extern "C" int scaffold_gibbs(int m,int k,const double* h,const double* gl,const double* refresh,
 double mutation,int burn,int retained,int chains,uint64_t seed,double* gp){
 if(m<1||k<1||burn<0||retained<1||chains<1||mutation<0||mutation>=.5)return 1;
 std::fill(gp,gp+3*m,0.);
 std::vector<double> l(3*m),x(size_t(m)*k),decay(m),effective;
 std::vector<int> mono(m),neutral(m),binary(m),index(m,-1),left(m),markers;int last=-1;double survive=1.;
 for(int t=0;t<m;++t){
  double top=std::max({gl[3*t],gl[3*t+1],gl[3*t+2]});for(int g=0;g<3;++g)l[3*t+g]=std::exp(std::max(-690.,gl[3*t+g]-top));
  neutral[t]=l[3*t]==l[3*t+1]&&l[3*t]==l[3*t+2];
  mono[t]=1;for(int a=0;a<k;++a){x[size_t(t)*k+a]=mutation+(1-2*mutation)*h[size_t(t)*k+a];if(h[size_t(t)*k+a]!=h[size_t(t)*k])mono[t]=0;}
  survive*=1-refresh[t];if(!mono[t]&&!neutral[t]){index[t]=int(markers.size());markers.push_back(t);effective.push_back(1-survive);last=index[t];survive=1.;}
  left[t]=last;decay[t]=survive;
 }
 std::vector<uint8_t> code(size_t(m)*k);
 for(int t=0;t<m;++t){binary[t]=1;for(int a=0;a<k;++a){double v=h[size_t(t)*k+a];if(v!=0.&&v!=1.)binary[t]=0;code[size_t(t)*k+a]=v==1.;}}
 HMMCache cache(size_t(markers.size())*k,markers.size(),size_t(m)*2);
 double* fw=nullptr;double* scale=nullptr;
 std::vector<double> back(k),conditional(k),scratch(k),mass(k);
 std::mt19937_64 rng(seed);std::uniform_real_distribution<double> uniform(0.,1.);
 auto draw=[&](const double* p){double total=0;for(int a=0;a<k;++a)total+=p[a];double u=uniform(rng)*total;for(int a=0;a<k-1;++a){u-=p[a];if(u<=0)return a;}return k-1;};
 auto value=[&](int t,int a){int ix=left[t];if(ix<0)return 1./k;return index[t]>=0?fw[size_t(ix)*k+a]:decay[t]*fw[size_t(ix)*k+a]+(1-decay[t])/k;};
 auto draw_fw=[&](int t){if(index[t]>=0)return draw(&fw[size_t(index[t])*k]);for(int a=0;a<k;++a)scratch[a]=value(t,a);return draw(scratch.data());};
 for(int chain=0;chain<chains;++chain){
  std::vector<int> allele[2]={std::vector<int>(m),std::vector<int>(m)};
  for(int c=0;c<2;++c){int donor=(2*chain+c)%k;for(int t=0;t<m;++t)allele[c][t]=uniform(rng)<x[size_t(t)*k+donor];}
  for(int iteration=0;iteration<burn+retained;++iteration)for(int chromosome=0;chromosome<2;++chromosome){
   const auto& other=allele[1-chromosome];
   std::string key;key.reserve(markers.size());for(int t:markers)key.push_back(char(other[t]));
   bool hit=false;auto& cached=cache.get(key,hit);fw=cached.fw.data();scale=cached.scale.data();
   double* posterior=cached.posterior.data();
   if(!hit)for(size_t ix=0;ix<markers.size();++ix){int t=markers[ix];double r=effective[ix],z=0;double* dest=&fw[ix*k];
    const double* row=&x[size_t(t)*k];double l0=l[3*t+other[t]],l1=l[3*t+other[t]+1];
    if(binary[t]){
     const uint8_t* codes=&code[size_t(t)*k];double table[2];
     for(int c=0;c<2;++c){double p=mutation+(1-2*mutation)*c;table[c]=(1-p)*l0+p*l1;}
     if(ix){const double* prev=dest-k;for(int a=0;a<k;++a)dest[a]=((1-r)*prev[a]+r/k)*table[codes[a]];}
     else for(int a=0;a<k;++a)dest[a]=(1./k)*table[codes[a]];
    }else{
     if(ix){const double* prev=dest-k;for(int a=0;a<k;++a){double p=row[a];dest[a]=((1-r)*prev[a]+r/k)*((1-p)*l0+p*l1);}}
     else for(int a=0;a<k;++a){double p=row[a];dest[a]=(1./k)*((1-p)*l0+p*l1);}
    }
    z=hmm_sum(dest,k);if(!(z>0)||!std::isfinite(z))return 2;scale[ix]=z;double inv_z=1./z;for(int a=0;a<k;++a)dest[a]*=inv_z;
   }
   if(iteration>=burn){std::fill(conditional.begin(),conditional.end(),1.);double right_decay=1.,right_mean=1.;bool have_right=false;
    for(int t=m-1;t>=0;--t){if(have_right)right_decay*=1-refresh[t+1];double l0=l[3*t+other[t]],l1=l[3*t+other[t]+1];
     if(mono[t]){double p=x[size_t(t)*k],v0=(1-p)*l0,v1=p*l1;gp[3*t+other[t]]+=v0/(v0+v1);gp[3*t+other[t]+1]+=v1/(v0+v1);continue;}
     if(cached.posterior_ready){gp[3*t+other[t]]+=posterior[2*t];gp[3*t+other[t]+1]+=posterior[2*t+1];continue;}
     if(neutral[t]){
      for(int a=0;a<k;++a)back[a]=right_decay*conditional[a]+(1-right_decay)*right_mean;double z=0,p1=0;
      for(int a=0;a<k;++a){double w=value(t,a)*back[a];z+=w;p1+=w*x[size_t(t)*k+a];}
      if(!(z>0))return 4;posterior[2*t]=(z-p1)/z;posterior[2*t+1]=p1/z;gp[3*t+other[t]]+=posterior[2*t];gp[3*t+other[t]+1]+=posterior[2*t+1];continue;
     }
     int ix=index[t];double p0=0,p1=0,z=0;const double* f=&fw[size_t(ix)*k];
     double inv_scale=1./scale[ix],next_mean=0.;bool fused=false;
     if(binary[t]){double table0[2],table1[2],emission[2];
      for(int c=0;c<2;++c){double p=mutation+(1-2*mutation)*c,v0=(1-p)*l0,v1=p*l1;emission[c]=v0+v1;table0[c]=v0/emission[c];table1[c]=v1/emission[c];}
      const uint8_t* codes=&code[size_t(t)*k];double buckets[256]={},table[256]={};table[0]=emission[0];table[1]=emission[1];
      fused=fused_backward(codes,f,conditional.data(),k,2,right_decay,right_mean,table,inv_scale,buckets,next_mean);
      if(!fused){
       for(int a=0;a<k;++a)back[a]=right_decay*conditional[a]+(1-right_decay)*right_mean;
       for(int a=0;a<k;++a){int c=codes[a];mass[a]=f[a]*back[a];conditional[a]=(back[a]*emission[c])*inv_scale;}
       allele_buckets(codes,mass.data(),k,2,buckets);
      }
      z=buckets[0]+buckets[1];p0=buckets[0]*table0[0]+buckets[1]*table0[1];p1=buckets[0]*table1[0]+buckets[1]*table1[1];
     }else{
      for(int a=0;a<k;++a)back[a]=right_decay*conditional[a]+(1-right_decay)*right_mean;
      for(int a=0;a<k;++a){double p=x[size_t(t)*k+a],v0=(1-p)*l0,v1=p*l1,w=f[a]*back[a];z+=w;p0+=w*v0/(v0+v1);p1+=w*v1/(v0+v1);conditional[a]=(back[a]*(v0+v1))*inv_scale;}
     }
     posterior[2*t]=p0/z;posterior[2*t+1]=p1/z;gp[3*t+other[t]]+=posterior[2*t];gp[3*t+other[t]+1]+=posterior[2*t+1];
     right_mean=fused?next_mean:hmm_sum(conditional.data(),k,double(k));right_decay=1.;have_right=true;
    }
   }
   if(iteration>=burn)cached.posterior_ready=true;
   int donor=draw_fw(m-1);
   for(int t=m-1;t>=0;--t){double p=x[size_t(t)*k+donor],v0=(1-p)*l[3*t+other[t]],v1=p*l[3*t+other[t]+1];allele[chromosome][t]=uniform(rng)<v1/(v0+v1);
    if(t){double stay=(1-refresh[t])*value(t-1,donor),change=refresh[t]/k;if(uniform(rng)*(stay+change)>=stay)donor=draw_fw(t-1);}
   }
  }
 }
 double expected=double(chains)*retained*2;
 for(int t=0;t<m;++t){double z=gp[3*t]+gp[3*t+1]+gp[3*t+2];if(!(z>0)||std::abs(z-expected)>expected*1e-5)return 3;for(int g=0;g<3;++g)gp[3*t+g]/=z;}
 return 0;
}
