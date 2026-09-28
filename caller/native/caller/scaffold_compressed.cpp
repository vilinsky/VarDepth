// Exact elimination of donor-invariant SNP emissions; original Gibbs RNG order.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
extern "C" int scaffold_gibbs(int m,int k,const double* h,const double* gl,const double* refresh,
 double mutation,int burn,int retained,int chains,uint64_t seed,double* gp){
 if(m<1||k<1||burn<0||retained<1||chains<1||mutation<0||mutation>=.5)return 1;
 std::fill(gp,gp+3*m,0.);
 std::vector<double> l(3*m),x(size_t(m)*k),decay(m),effective;
 std::vector<int> mono(m),index(m,-1),left(m),markers;int last=-1;double survive=1.;
 for(int t=0;t<m;++t){
  double top=std::max({gl[3*t],gl[3*t+1],gl[3*t+2]});for(int g=0;g<3;++g)l[3*t+g]=std::exp(std::max(-690.,gl[3*t+g]-top));
  mono[t]=1;for(int a=0;a<k;++a){x[size_t(t)*k+a]=mutation+(1-2*mutation)*h[size_t(t)*k+a];if(h[size_t(t)*k+a]!=h[size_t(t)*k])mono[t]=0;}
  survive*=1-refresh[t];if(!mono[t]){index[t]=int(markers.size());markers.push_back(t);effective.push_back(1-survive);last=index[t];survive=1.;}
  left[t]=last;decay[t]=survive;
 }
 std::vector<double> fw(size_t(markers.size())*k),scale(markers.size()),back(k),conditional(k),scratch(k);
 std::mt19937_64 rng(seed);std::uniform_real_distribution<double> uniform(0.,1.);
 auto draw=[&](const double* p){double total=0;for(int a=0;a<k;++a)total+=p[a];double u=uniform(rng)*total;for(int a=0;a<k-1;++a){u-=p[a];if(u<=0)return a;}return k-1;};
 auto value=[&](int t,int a){int ix=left[t];if(ix<0)return 1./k;return index[t]>=0?fw[size_t(ix)*k+a]:decay[t]*fw[size_t(ix)*k+a]+(1-decay[t])/k;};
 auto draw_fw=[&](int t){if(index[t]>=0)return draw(&fw[size_t(index[t])*k]);for(int a=0;a<k;++a)scratch[a]=value(t,a);return draw(scratch.data());};
 for(int chain=0;chain<chains;++chain){
  std::vector<int> allele[2]={std::vector<int>(m),std::vector<int>(m)};
  for(int c=0;c<2;++c){int donor=(2*chain+c)%k;for(int t=0;t<m;++t)allele[c][t]=uniform(rng)<x[size_t(t)*k+donor];}
  for(int iteration=0;iteration<burn+retained;++iteration)for(int chromosome=0;chromosome<2;++chromosome){
   const auto& other=allele[1-chromosome];
   for(size_t ix=0;ix<markers.size();++ix){int t=markers[ix];double r=effective[ix],z=0;double* dest=&fw[ix*k];
    const double* row=&x[size_t(t)*k];double l0=l[3*t+other[t]],l1=l[3*t+other[t]+1];
    if(ix){const double* prev=dest-k;for(int a=0;a<k;++a){double p=row[a];dest[a]=((1-r)*prev[a]+r/k)*((1-p)*l0+p*l1);}}
    else for(int a=0;a<k;++a){double p=row[a];dest[a]=(1./k)*((1-p)*l0+p*l1);}
    for(int a=0;a<k;++a)z+=dest[a];if(!(z>0)||!std::isfinite(z))return 2;scale[ix]=z;for(int a=0;a<k;++a)dest[a]/=z;
   }
   if(iteration>=burn){std::fill(back.begin(),back.end(),1.);
    for(int t=m-1;t>=0;--t){double l0=l[3*t+other[t]],l1=l[3*t+other[t]+1];
     if(mono[t]){double p=x[size_t(t)*k],v0=(1-p)*l0,v1=p*l1;gp[3*t+other[t]]+=v0/(v0+v1);gp[3*t+other[t]+1]+=v1/(v0+v1);continue;}
     int ix=index[t];double p0=0,p1=0,z=0;const double* f=&fw[size_t(ix)*k];
     for(int a=0;a<k;++a){double p=x[size_t(t)*k+a],v0=(1-p)*l0,v1=p*l1,w=f[a]*back[a];z+=w;p0+=w*v0/(v0+v1);p1+=w*v1/(v0+v1);conditional[a]=back[a]*(v0+v1)/scale[ix];}
     gp[3*t+other[t]]+=p0/z;gp[3*t+other[t]+1]+=p1/z;
     if(ix){double mean=0;for(int a=0;a<k;++a)mean+=conditional[a]/k;for(int a=0;a<k;++a)back[a]=(1-effective[ix])*conditional[a]+effective[ix]*mean;}
    }
   }
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
