// Independent categorical Li-Stephens conditional-copying implementation.
// One block is one physical local haplotype, including SNPs and short indels.
// The two chromosomes are alternately sampled; read GL is used once per block.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

static int gi(int a, int b) { if (a > b) std::swap(a,b); return b*(b+1)/2+a; }
extern "C" int categorical_gibbs(int m, int k, const uint8_t* hap,
 const int32_t* counts, const int64_t* offsets, const double* gl,
 const double* refresh, double mutation, int burn, int retained, int chains,
 uint64_t seed, double* gp) {
 if (m<1 || k<1 || burn<0 || retained<1 || chains<1 || mutation<0 || mutation>=1) return 1;
 const size_t total=offsets[m];
 std::fill(gp,gp+total,0.);
 std::vector<double> l(total), fw(size_t(m)*k), scale(m), back(k), conditional(k), site(24), weights(24);
 std::vector<double> emission_buffer(k), mass(k), divided(k);
 for(int t=0;t<m;++t) {
  int n=counts[t]; if(n<2 || n>24 || offsets[t+1]-offsets[t]!=n*(n+1)/2) return 2;
  double top=*std::max_element(gl+offsets[t],gl+offsets[t+1]);
  for(int64_t j=offsets[t];j<offsets[t+1];++j) l[j]=std::exp(std::max(-690.,gl[j]-top));
  for(int a=0;a<k;++a) if(hap[size_t(t)*k+a]>=n && hap[size_t(t)*k+a]!=255) return 3;
 }
 std::mt19937_64 rng(seed); std::uniform_real_distribution<double> uniform(0.,1.);
 auto draw=[&](const double* p,int n) { double z=0;for(int j=0;j<n;++j)z+=p[j];
  double u=uniform(rng)*z;for(int j=0;j<n-1;++j){u-=p[j];if(u<=0)return j;}return n-1; };
 for(int chain=0;chain<chains;++chain) {
  std::vector<int> allele[2]={std::vector<int>(m),std::vector<int>(m)};
  for(int c=0;c<2;++c)for(int t=0;t<m;++t) {
   int donor=hap[size_t(t)*k+(2*chain+c)%k],n=counts[t];
   for(int j=0;j<n;++j)weights[j]=donor==255 ? 1./n : mutation/n+(j==donor ? 1-mutation : 0);
   allele[c][t]=draw(weights.data(),n);
  }
  for(int iteration=0;iteration<burn+retained;++iteration)for(int chromosome=0;chromosome<2;++chromosome) {
   const auto& other=allele[1-chromosome];
   auto local=[&](int t) { int n=counts[t];double mean=0;
    for(int a=0;a<n;++a) {site[a]=l[offsets[t]+gi(a,other[t])];mean+=site[a]/n;}return mean; };
   for(int t=0;t<m;++t) {
    double mean=local(t),z=0,r=refresh[t];const double* prev=t ? &fw[size_t(t-1)*k] : nullptr;
    double table[256];table[255]=mean;
    for(int j=0;j<counts[t];++j)table[j]=(1-mutation)*site[j]+mutation*mean;
    for(int a=0;a<k;++a)emission_buffer[a]=table[hap[size_t(t)*k+a]];
    double* dest=&fw[size_t(t)*k];
    if(t)for(int a=0;a<k;++a)dest[a]=((1-r)*prev[a]+r/k)*emission_buffer[a];
    else for(int a=0;a<k;++a)dest[a]=(1./k)*emission_buffer[a];
    // Keep the original summation order; only independent donor work is SIMD.
    #pragma omp simd reduction(+:z)
    for(int a=0;a<k;++a)z+=dest[a];
    if(!(z>0) || !std::isfinite(z))return 4;scale[t]=z;
    for(int a=0;a<k;++a)fw[size_t(t)*k+a]/=z;
   }
   if(iteration>=burn) {
    std::fill(back.begin(),back.end(),1.);
    for(int t=m-1;t>=0;--t) {
     int n=counts[t];double mean=local(t),z=0,uniform_mass=0;
     std::fill(weights.begin(),weights.end(),0.);
     double table[256];table[255]=mean;
     for(int j=0;j<n;++j)table[j]=(1-mutation)*site[j]+mutation*mean;
     if(n<=4){
      // Collapse equal categorical emissions only inside the posterior sum.
      // Keep every donor state, transition, Gibbs draw and iteration.
      double b0=0,b1=0,b2=0,b3=0,bu=0;
      const uint8_t* hrow=&hap[size_t(t)*k];const double* frow=&fw[size_t(t)*k];
      const double invscale=1./scale[t];
      #pragma omp simd reduction(+:b0,b1,b2,b3,bu)
      for(int a=0;a<k;++a){
       int d=hrow[a];double w=frow[a]*back[a];
       b0+=d==0?w:0.;b1+=d==1?w:0.;b2+=d==2?w:0.;b3+=d==3?w:0.;bu+=d==255?w:0.;
       conditional[a]=back[a]*table[d]*invscale;
      }
      double buckets[4]={b0,b1,b2,b3};z=bu+b0+b1+b2+b3;
      uniform_mass=bu/table[255]/n;
      for(int j=0;j<n;++j){double divided=buckets[j]/table[j];uniform_mass+=divided*mutation/n;weights[j]=divided*(1-mutation);}
     }else{
     for(int a=0;a<k;++a)emission_buffer[a]=table[hap[size_t(t)*k+a]];
     for(int a=0;a<k;++a){
      mass[a]=fw[size_t(t)*k+a]*back[a];
      divided[a]=mass[a]/emission_buffer[a];
      conditional[a]=back[a]*emission_buffer[a]/scale[t];
     }
     for(int a=0;a<k;++a) {
      int donor=hap[size_t(t)*k+a];
      z+=mass[a];
      if(donor==255)uniform_mass+=divided[a]/n;
      else {uniform_mass+=divided[a]*mutation/n;weights[donor]+=divided[a]*(1-mutation);}
     }
     }
     if(!(z>0) || !std::isfinite(z))return 5;
     for(int a=0;a<n;++a)gp[offsets[t]+gi(a,other[t])]+=(uniform_mass+weights[a])*site[a]/z;
     if(t) {double average=0;
      #pragma omp simd reduction(+:average)
      for(int a=0;a<k;++a)average+=conditional[a]/k;
      for(int a=0;a<k;++a)back[a]=(1-refresh[t])*conditional[a]+refresh[t]*average;
     }
    }
   }
   int donor=draw(&fw[size_t(m-1)*k],k);
   for(int t=m-1;t>=0;--t) {
    local(t);int n=counts[t],value=hap[size_t(t)*k+donor];
    for(int a=0;a<n;++a)weights[a]=site[a]*(value==255 ? 1./n : mutation/n+(a==value ? 1-mutation : 0));
    allele[chromosome][t]=draw(weights.data(),n);
    if(t) {const double* prev=&fw[size_t(t-1)*k];double stay=(1-refresh[t])*prev[donor],change=refresh[t]/k;
     if(uniform(rng)*(stay+change)>=stay)donor=draw(prev,k);
    }
   }
  }
 }
 double expected=double(chains)*retained*2;
 for(int t=0;t<m;++t) {
  double z=0;for(int64_t j=offsets[t];j<offsets[t+1];++j)z+=gp[j];
  if(!(z>0) || std::abs(z-expected)>expected*1e-5)return 6;
  for(int64_t j=offsets[t];j<offsets[t+1];++j)gp[j]/=z;
 }
 return 0;
}
