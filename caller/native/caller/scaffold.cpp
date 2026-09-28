// Independently implemented conditional haploid copying sampler.
// Mathematical basis: Li-Stephens copying HMM; alternating conditional updates
// of two latent chromosomes. No external tool implementation is incorporated.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
using Vec=std::vector<double>;
extern "C" int scaffold_gibbs(int m,int k,const double* h,const double* gl,
 const double* refresh,double mutation,int burn,int retained,int chains,
 uint64_t seed,double* gp){
 if(m<1||k<1||burn<0||retained<1||chains<1||mutation<0||mutation>=.5)return 1;
 std::fill(gp,gp+3*m,0.);
 Vec l(3*m),x(size_t(m)*k),fw(size_t(m)*k),scale(m),back(k),conditional(k),mass(k),part0(k),part1(k);
 for(int t=0;t<m;++t){double top=std::max({gl[3*t],gl[3*t+1],gl[3*t+2]});
  for(int g=0;g<3;++g)l[3*t+g]=std::exp(std::max(-690.,gl[3*t+g]-top));
  for(int a=0;a<k;++a)x[size_t(t)*k+a]=mutation+(1-2*mutation)*h[size_t(t)*k+a];
 }
 std::mt19937_64 rng(seed);std::uniform_real_distribution<double> uniform(0.,1.);
 auto draw=[&](const double* p,int n){double total=0;for(int j=0;j<n;++j)total+=p[j];
  double u=uniform(rng)*total;for(int j=0;j<n-1;++j){u-=p[j];if(u<=0)return j;}return n-1;};
 for(int chain=0;chain<chains;++chain){
  std::vector<int> allele[2]={std::vector<int>(m),std::vector<int>(m)};
  for(int c=0;c<2;++c){int donor=(2*chain+c)%k;
   for(int t=0;t<m;++t)allele[c][t]=uniform(rng)<x[size_t(t)*k+donor];}
  for(int iteration=0;iteration<burn+retained;++iteration)for(int chromosome=0;chromosome<2;++chromosome){
   const auto& other=allele[1-chromosome];
   for(int t=0;t<m;++t){double z=0,r=refresh[t];const double* prev=t?&fw[size_t(t-1)*k]:nullptr;
    double* dest=&fw[size_t(t)*k];const double* row=&x[size_t(t)*k];
    if(t)for(int a=0;a<k;++a){double p=row[a];double e=(1-p)*l[3*t+other[t]]+p*l[3*t+other[t]+1];
     dest[a]=((1-r)*prev[a]+r/k)*e;}
    else for(int a=0;a<k;++a){double p=row[a];double e=(1-p)*l[3*t+other[t]]+p*l[3*t+other[t]+1];dest[a]=(1./k)*e;}
    #pragma omp simd reduction(+:z)
    for(int a=0;a<k;++a)z+=dest[a];
    if(!(z>0)||!std::isfinite(z))return 2;scale[t]=z;
    for(int a=0;a<k;++a)fw[size_t(t)*k+a]/=z;
   }
   if(iteration>=burn){
    std::fill(back.begin(),back.end(),1.);
    for(int t=m-1;t>=0;--t){double p0=0,p1=0,z=0;
     for(int a=0;a<k;++a){double p=x[size_t(t)*k+a];double v0=(1-p)*l[3*t+other[t]],v1=p*l[3*t+other[t]+1];
      double w=fw[size_t(t)*k+a]*back[a];mass[a]=w;part0[a]=w*v0/(v0+v1);part1[a]=w*v1/(v0+v1);
      conditional[a]=back[a]*(v0+v1)/scale[t];}
     #pragma omp simd reduction(+:z,p0,p1)
     for(int a=0;a<k;++a){z+=mass[a];p0+=part0[a];p1+=part1[a];}
     gp[3*t+other[t]]+=p0/z;gp[3*t+other[t]+1]+=p1/z;
     if(t){double mean=0;
      #pragma omp simd reduction(+:mean)
      for(int a=0;a<k;++a)mean+=conditional[a]/k;
      for(int a=0;a<k;++a)back[a]=(1-refresh[t])*conditional[a]+refresh[t]*mean;}
    }
   }
   // Backward sample the copying path, then each emitted latent allele.
   int donor=draw(&fw[size_t(m-1)*k],k);
   for(int t=m-1;t>=0;--t){double p=x[size_t(t)*k+donor];double v0=(1-p)*l[3*t+other[t]],v1=p*l[3*t+other[t]+1];
    allele[chromosome][t]=uniform(rng)<v1/(v0+v1);
    if(t){const double* prev=&fw[size_t(t-1)*k];double stay=(1-refresh[t])*prev[donor],change=refresh[t]/k;
     if(uniform(rng)*(stay+change)>=stay)donor=draw(prev,k);}
   }
  }
 }
 double denominator=double(chains)*retained*2;
 for(int t=0;t<m;++t){double z=gp[3*t]+gp[3*t+1]+gp[3*t+2];
  if(!(z>0)||std::abs(z-denominator)>denominator*1e-5)return 3;
  for(int g=0;g<3;++g)gp[3*t+g]/=z;}
 return 0;
}
