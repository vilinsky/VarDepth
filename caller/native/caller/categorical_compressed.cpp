// Exact algebraic elimination of donor-invariant emissions in a copying HMM.
// All markers, donors, retained Gibbs draws and allele likelihoods remain.
// Floating-point operation order differs; numerical/public parity is required.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
static int gi(int a,int b){if(a>b)std::swap(a,b);return b*(b+1)/2+a;}
extern "C" int categorical_gibbs(int m,int k,const uint8_t* hap,const int32_t* counts,
 const int64_t* offsets,const double* gl,const double* refresh,double mutation,
 int burn,int retained,int chains,uint64_t seed,double* gp){
 if(m<1||k<1||burn<0||retained<1||chains<1||mutation<0||mutation>=1)return 1;
 size_t total=offsets[m];std::fill(gp,gp+total,0.);
 std::vector<double> likelihood(total),decay(m),effective,scale;
 std::vector<int> mono(m),left(m),index(m,-1),markers;
 int last=-1;double survive=1.;
 for(int t=0;t<m;++t){
  int n=counts[t];if(n<2||n>24||offsets[t+1]-offsets[t]!=n*(n+1)/2)return 2;
  double top=*std::max_element(gl+offsets[t],gl+offsets[t+1]);
  for(int64_t j=offsets[t];j<offsets[t+1];++j)likelihood[j]=std::exp(std::max(-690.,gl[j]-top));
  mono[t]=hap[size_t(t)*k];
  for(int a=0;a<k;++a){int d=hap[size_t(t)*k+a];if(d>=n&&d!=255)return 3;if(d!=hap[size_t(t)*k])mono[t]=-1;}
  survive*=1-refresh[t];
  if(mono[t]<0){index[t]=int(markers.size());markers.push_back(t);effective.push_back(1-survive);last=index[t];survive=1.;}
  left[t]=last;decay[t]=survive;
 }
 std::vector<double> fw(size_t(markers.size())*k),back(k),conditional(k),scratch(k),mass(k),emission(k);
 scale.resize(markers.size());
 std::mt19937_64 rng(seed);std::uniform_real_distribution<double> uniform(0.,1.);
 auto draw=[&](const double* p,int n){double z=0;for(int j=0;j<n;++j)z+=p[j];double u=uniform(rng)*z;
  for(int j=0;j<n-1;++j){u-=p[j];if(u<=0)return j;}return n-1;};
 auto fw_value=[&](int t,int a){int pos=left[t];if(pos<0)return 1./k;
  return index[t]>=0?fw[size_t(pos)*k+a]:decay[t]*fw[size_t(pos)*k+a]+(1-decay[t])/k;};
 auto draw_fw=[&](int t){if(index[t]>=0)return draw(&fw[size_t(index[t])*k],k);
  for(int a=0;a<k;++a)scratch[a]=fw_value(t,a);return draw(scratch.data(),k);};
 for(int chain=0;chain<chains;++chain){
  std::vector<int> allele[2]={std::vector<int>(m),std::vector<int>(m)};
  double weights[24],site[24];
  for(int c=0;c<2;++c)for(int t=0;t<m;++t){int d=hap[size_t(t)*k+(2*chain+c)%k],n=counts[t];
   for(int j=0;j<n;++j)weights[j]=d==255?1./n:mutation/n+(j==d?1-mutation:0);allele[c][t]=draw(weights,n);}
  for(int iteration=0;iteration<burn+retained;++iteration)for(int chromosome=0;chromosome<2;++chromosome){
   const auto& other=allele[1-chromosome];
   auto local=[&](int t){int n=counts[t];double mean=0;for(int a=0;a<n;++a){site[a]=likelihood[offsets[t]+gi(a,other[t])];mean+=site[a]/n;}return mean;};
   for(size_t ix=0;ix<markers.size();++ix){
    int t=markers[ix],n=counts[t];double mean=local(t),table[256];table[255]=mean;
    for(int j=0;j<n;++j)table[j]=(1-mutation)*site[j]+mutation*mean;
    const uint8_t* h=&hap[size_t(t)*k];double* dest=&fw[ix*k];double r=effective[ix];
    for(int a=0;a<k;++a)emission[a]=table[h[a]];
    if(ix){const double* prev=dest-k;for(int a=0;a<k;++a)dest[a]=((1-r)*prev[a]+r/k)*emission[a];}
    else for(int a=0;a<k;++a)dest[a]=emission[a]/k;
    double z=0;
    #pragma omp simd reduction(+:z)
    for(int a=0;a<k;++a)z+=dest[a];
    if(!(z>0)||!std::isfinite(z))return 4;scale[ix]=z;
    for(int a=0;a<k;++a)dest[a]/=z;
   }
   if(iteration>=burn){
    std::fill(back.begin(),back.end(),1.);
    for(int t=m-1;t>=0;--t){
     int n=counts[t];double mean=local(t),z=0;
     if(mono[t]>=0){
      int d=mono[t];
      for(int a=0;a<n;++a){weights[a]=site[a]*(d==255?1./n:mutation/n+(a==d?1-mutation:0));z+=weights[a];}
      if(!(z>0)){std::fprintf(stderr,"mono failure t=%d z=%g\n",t,z);return 5;}
      for(int a=0;a<n;++a)gp[offsets[t]+gi(a,other[t])]+=weights[a]/z;
      continue;
     }
     int ix=index[t];const uint8_t* h=&hap[size_t(t)*k];const double* f=&fw[size_t(ix)*k];
     double table[256];table[255]=mean;for(int j=0;j<n;++j)table[j]=(1-mutation)*site[j]+mutation*mean;
     double buckets[256]={};
     for(int a=0;a<k;++a)emission[a]=table[h[a]];
     for(int a=0;a<k;++a){mass[a]=f[a]*back[a];conditional[a]=back[a]*emission[a]/scale[ix];}
     if(n<=4){double b0=0,b1=0,b2=0,b3=0,bu=0;
      #pragma omp simd reduction(+:b0,b1,b2,b3,bu)
      for(int a=0;a<k;++a){int d=h[a];double w=mass[a];b0+=d==0?w:0.;b1+=d==1?w:0.;b2+=d==2?w:0.;b3+=d==3?w:0.;bu+=d==255?w:0.;}
      buckets[0]=b0;buckets[1]=b1;buckets[2]=b2;buckets[3]=b3;buckets[255]=bu;
     }else for(int a=0;a<k;++a)buckets[h[a]]+=mass[a];
     z=buckets[255];double uniform_mass=buckets[255]/table[255]/n;
     for(int a=0;a<n;++a){z+=buckets[a];double divided=buckets[a]/table[a];uniform_mass+=divided*mutation/n;weights[a]=divided*(1-mutation);}
     if(!(z>0)||!std::isfinite(z)){std::fprintf(stderr,"posterior failure t=%d ix=%d z=%g backmax=%g scale=%g mean=%g iter=%d\n",t,ix,z,*std::max_element(back.begin(),back.end()),scale[ix],mean,iteration);return 5;}
     for(int a=0;a<n;++a)gp[offsets[t]+gi(a,other[t])]+=(uniform_mass+weights[a])*site[a]/z;
     if(ix){double average=0;
      #pragma omp simd reduction(+:average)
      for(int a=0;a<k;++a)average+=conditional[a]/k;
      for(int a=0;a<k;++a)back[a]=(1-effective[ix])*conditional[a]+effective[ix]*average;
     }
    }
   }
   int donor=draw_fw(m-1);
   for(int t=m-1;t>=0;--t){local(t);int n=counts[t],d=hap[size_t(t)*k+donor];
    for(int a=0;a<n;++a)weights[a]=site[a]*(d==255?1./n:mutation/n+(a==d?1-mutation:0));
    allele[chromosome][t]=draw(weights,n);
    if(t){double stay=(1-refresh[t])*fw_value(t-1,donor),change=refresh[t]/k;
     if(uniform(rng)*(stay+change)>=stay)donor=draw_fw(t-1);}
   }
  }
 }
 double expected=double(chains)*retained*2;
 for(int t=0;t<m;++t){double z=0;for(int64_t j=offsets[t];j<offsets[t+1];++j)z+=gp[j];
  if(!(z>0)||std::abs(z-expected)>expected*1e-5)return 6;
  for(int64_t j=offsets[t];j<offsets[t+1];++j)gp[j]/=z;}
 return 0;
}
