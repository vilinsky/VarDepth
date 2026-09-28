// Independent CPU batching of the existing double-precision PairHMM.
// Lanes represent independent read/haplotype pairs. No pruning or banding.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <new>
struct Aligned {
 void* p=nullptr; size_t bytes=0;
 ~Aligned(){if(p)::operator delete(p,std::align_val_t(64));}
 void reserve(size_t n){if(n>bytes){if(p)::operator delete(p,std::align_val_t(64));p=::operator new(n,std::align_val_t(64));bytes=n;}}
};
static const double* errors(){static const std::vector<double> e=[](){std::vector<double> v(256);for(int i=0;i<256;++i)v[i]=std::pow(10.,-std::min(i,60)/10.);return v;}();return e.data();}
#define SELECT(mask,a,b) ((Q)(((I)(a)&(mask))|((I)(b)&~(mask))))

 static void batch_2(const char* const* reads,const unsigned char* const* quals,const int* rn,
 const char* const* haps,const int* hn,int count,int mode,double* output){
 using Q=double __attribute__((vector_size(16)));
 using I=long long __attribute__((vector_size(16)));
 thread_local Aligned buffer;
 const double* err=errors();
 for(int offset=0;offset<count;offset+=2){
  int take=std::min(2,count-offset),n=1,m=1;
  for(int z=0;z<take;++z){n=std::max(n,rn[offset+z]);m=std::max(m,hn[offset+z]);}
  buffer.reserve((size_t(8)*(m+1)+size_t(2)*(n+1))*sizeof(Q));
  Q* pm=(Q*)buffer.p;Q* pi=pm+m+1;Q* pd=pi+m+1;Q* cm=pd+m+1;Q* ci=cm+m+1;Q* cd=ci+m+1;
  I* codes=(I*)(cd+m+1);I* active=codes+m+1;I* bases=active+m+1;Q* error=(Q*)(bases+n+1);
  double logs[2]={};
  Q zero={};Q quarter=zero+.25,g=zero+.001,x=zero+.1,match=zero+(1-2*.001),exit_gap=zero+(1-.1);
  for(int j=0;j<=m;++j){
   double deletion=j? .001*std::pow(.1,j-1):0.;
   for(int z=0;z<2;++z){int idx=offset+z,ml=z<take?hn[idx]:1;
    bool valid=z<take&&rn[idx]>0&&ml>0;
    active[j][z]=j<=(valid?ml:1)?-1LL:0LL;
    codes[j][z]=valid&&j>0&&j<=ml?(unsigned char)haps[idx][j-1]:0;
    pm[j][z]=mode==1?(j==0?1.:0.):(j<=(valid?ml:1)?1./((valid?ml:1)+1):0.);
    pi[j][z]=0.;pd[j][z]=mode==1&&active[j][z]?deletion:0.;
   }
  }
  for(int i=1;i<=n;++i)for(int z=0;z<2;++z){int idx=offset+z;bool valid=z<take&&rn[idx]>=i&&hn[idx]>0;
   bases[i][z]=valid?(unsigned char)reads[idx][i-1]:0;error[i][z]=valid?err[quals[idx][i-1]]:.001;}
  for(int i=1;i<=n;++i){
   cm[0]=cd[0]=zero;ci[0]=quarter*(g*pm[0]+x*pi[0]);
   Q scale=cm[0]+ci[0]+cd[0],good=(zero+1.)-error[i],bad=error[i]/3.;
   for(int j=1;j<=m;++j){
    Q emit=SELECT(bases[i]==codes[j],good,bad);
    cm[j]=SELECT(active[j],emit*(match*pm[j-1]+exit_gap*(pi[j-1]+pd[j-1])),zero);
    ci[j]=SELECT(active[j],quarter*(g*pm[j]+x*pi[j]),zero);
    cd[j]=SELECT(active[j],g*cm[j-1]+x*cd[j-1],zero);
    Q sum=cm[j]+ci[j]+cd[j];scale=SELECT(sum>scale,sum,scale);
   }
   if(mode==2){
    Q factor=zero+1.;bool low=false;
    for(int z=0;z<2;++z)if(scale[z]<0x1p-450){factor[z]=0x1p512;logs[z]-=512*std::log(2.);low=true;}
    if(low)for(int j=0;j<=m;++j){cm[j]*=factor;ci[j]*=factor;cd[j]*=factor;}
   }else{
    for(int z=0;z<2;++z){if(!(scale[z]>0))scale[z]=1.;logs[z]+=std::log(scale[z]);}
    for(int j=0;j<=m;++j){cm[j]/=scale;ci[j]/=scale;cd[j]/=scale;}
   }
   std::swap(pm,cm);std::swap(pi,ci);std::swap(pd,cd);
   bool finish=false;for(int z=0;z<take;++z)if(rn[offset+z]==i)finish=true;
   if(finish){
    Q sum=zero;if(mode!=1)for(int j=0;j<=m;++j)sum+=pm[j]+pi[j];
    for(int z=0;z<take;++z)if(rn[offset+z]==i){int ml=hn[offset+z];
     double v=mode==1&&ml>0?pm[ml][z]+pi[ml][z]+pd[ml][z]:sum[z];
     output[offset+z]=std::log(std::max(1e-300,v))+logs[z];}
   }
  }
  for(int z=0;z<take;++z)if(rn[offset+z]<=0||hn[offset+z]<=0)output[offset+z]=-INFINITY;
 }
}
#if defined(__x86_64__)

__attribute__((target("avx2"))) static void batch_4(const char* const* reads,const unsigned char* const* quals,const int* rn,
 const char* const* haps,const int* hn,int count,int mode,double* output){
 using Q=double __attribute__((vector_size(32)));
 using I=long long __attribute__((vector_size(32)));
 thread_local Aligned buffer;
 const double* err=errors();
 for(int offset=0;offset<count;offset+=4){
  int take=std::min(4,count-offset),n=1,m=1;
  for(int z=0;z<take;++z){n=std::max(n,rn[offset+z]);m=std::max(m,hn[offset+z]);}
  buffer.reserve((size_t(8)*(m+1)+size_t(2)*(n+1))*sizeof(Q));
  Q* pm=(Q*)buffer.p;Q* pi=pm+m+1;Q* pd=pi+m+1;Q* cm=pd+m+1;Q* ci=cm+m+1;Q* cd=ci+m+1;
  I* codes=(I*)(cd+m+1);I* active=codes+m+1;I* bases=active+m+1;Q* error=(Q*)(bases+n+1);
  double logs[4]={};
  Q zero={};Q quarter=zero+.25,g=zero+.001,x=zero+.1,match=zero+(1-2*.001),exit_gap=zero+(1-.1);
  for(int j=0;j<=m;++j){
   double deletion=j? .001*std::pow(.1,j-1):0.;
   for(int z=0;z<4;++z){int idx=offset+z,ml=z<take?hn[idx]:1;
    bool valid=z<take&&rn[idx]>0&&ml>0;
    active[j][z]=j<=(valid?ml:1)?-1LL:0LL;
    codes[j][z]=valid&&j>0&&j<=ml?(unsigned char)haps[idx][j-1]:0;
    pm[j][z]=mode==1?(j==0?1.:0.):(j<=(valid?ml:1)?1./((valid?ml:1)+1):0.);
    pi[j][z]=0.;pd[j][z]=mode==1&&active[j][z]?deletion:0.;
   }
  }
  for(int i=1;i<=n;++i)for(int z=0;z<4;++z){int idx=offset+z;bool valid=z<take&&rn[idx]>=i&&hn[idx]>0;
   bases[i][z]=valid?(unsigned char)reads[idx][i-1]:0;error[i][z]=valid?err[quals[idx][i-1]]:.001;}
  for(int i=1;i<=n;++i){
   cm[0]=cd[0]=zero;ci[0]=quarter*(g*pm[0]+x*pi[0]);
   Q scale=cm[0]+ci[0]+cd[0],good=(zero+1.)-error[i],bad=error[i]/3.;
   for(int j=1;j<=m;++j){
    Q emit=SELECT(bases[i]==codes[j],good,bad);
    cm[j]=SELECT(active[j],emit*(match*pm[j-1]+exit_gap*(pi[j-1]+pd[j-1])),zero);
    ci[j]=SELECT(active[j],quarter*(g*pm[j]+x*pi[j]),zero);
    cd[j]=SELECT(active[j],g*cm[j-1]+x*cd[j-1],zero);
    Q sum=cm[j]+ci[j]+cd[j];scale=SELECT(sum>scale,sum,scale);
   }
   if(mode==2){
    Q factor=zero+1.;bool low=false;
    for(int z=0;z<4;++z)if(scale[z]<0x1p-450){factor[z]=0x1p512;logs[z]-=512*std::log(2.);low=true;}
    if(low)for(int j=0;j<=m;++j){cm[j]*=factor;ci[j]*=factor;cd[j]*=factor;}
   }else{
    for(int z=0;z<4;++z){if(!(scale[z]>0))scale[z]=1.;logs[z]+=std::log(scale[z]);}
    for(int j=0;j<=m;++j){cm[j]/=scale;ci[j]/=scale;cd[j]/=scale;}
   }
   std::swap(pm,cm);std::swap(pi,ci);std::swap(pd,cd);
   bool finish=false;for(int z=0;z<take;++z)if(rn[offset+z]==i)finish=true;
   if(finish){
    Q sum=zero;if(mode!=1)for(int j=0;j<=m;++j)sum+=pm[j]+pi[j];
    for(int z=0;z<take;++z)if(rn[offset+z]==i){int ml=hn[offset+z];
     double v=mode==1&&ml>0?pm[ml][z]+pi[ml][z]+pd[ml][z]:sum[z];
     output[offset+z]=std::log(std::max(1e-300,v))+logs[z];}
   }
  }
  for(int z=0;z<take;++z)if(rn[offset+z]<=0||hn[offset+z]<=0)output[offset+z]=-INFINITY;
 }
}
#endif
#if defined(__x86_64__)

__attribute__((target("avx512f"))) static void batch_8(const char* const* reads,const unsigned char* const* quals,const int* rn,
 const char* const* haps,const int* hn,int count,int mode,double* output){
 using Q=double __attribute__((vector_size(64)));
 using I=long long __attribute__((vector_size(64)));
 thread_local Aligned buffer;
 const double* err=errors();
 for(int offset=0;offset<count;offset+=8){
  int take=std::min(8,count-offset),n=1,m=1;
  for(int z=0;z<take;++z){n=std::max(n,rn[offset+z]);m=std::max(m,hn[offset+z]);}
  buffer.reserve((size_t(8)*(m+1)+size_t(2)*(n+1))*sizeof(Q));
  Q* pm=(Q*)buffer.p;Q* pi=pm+m+1;Q* pd=pi+m+1;Q* cm=pd+m+1;Q* ci=cm+m+1;Q* cd=ci+m+1;
  I* codes=(I*)(cd+m+1);I* active=codes+m+1;I* bases=active+m+1;Q* error=(Q*)(bases+n+1);
  double logs[8]={};
  Q zero={};Q quarter=zero+.25,g=zero+.001,x=zero+.1,match=zero+(1-2*.001),exit_gap=zero+(1-.1);
  for(int j=0;j<=m;++j){
   double deletion=j? .001*std::pow(.1,j-1):0.;
   for(int z=0;z<8;++z){int idx=offset+z,ml=z<take?hn[idx]:1;
    bool valid=z<take&&rn[idx]>0&&ml>0;
    active[j][z]=j<=(valid?ml:1)?-1LL:0LL;
    codes[j][z]=valid&&j>0&&j<=ml?(unsigned char)haps[idx][j-1]:0;
    pm[j][z]=mode==1?(j==0?1.:0.):(j<=(valid?ml:1)?1./((valid?ml:1)+1):0.);
    pi[j][z]=0.;pd[j][z]=mode==1&&active[j][z]?deletion:0.;
   }
  }
  for(int i=1;i<=n;++i)for(int z=0;z<8;++z){int idx=offset+z;bool valid=z<take&&rn[idx]>=i&&hn[idx]>0;
   bases[i][z]=valid?(unsigned char)reads[idx][i-1]:0;error[i][z]=valid?err[quals[idx][i-1]]:.001;}
  for(int i=1;i<=n;++i){
   cm[0]=cd[0]=zero;ci[0]=quarter*(g*pm[0]+x*pi[0]);
   Q scale=cm[0]+ci[0]+cd[0],good=(zero+1.)-error[i],bad=error[i]/3.;
   for(int j=1;j<=m;++j){
    Q emit=SELECT(bases[i]==codes[j],good,bad);
    cm[j]=SELECT(active[j],emit*(match*pm[j-1]+exit_gap*(pi[j-1]+pd[j-1])),zero);
    ci[j]=SELECT(active[j],quarter*(g*pm[j]+x*pi[j]),zero);
    cd[j]=SELECT(active[j],g*cm[j-1]+x*cd[j-1],zero);
    Q sum=cm[j]+ci[j]+cd[j];scale=SELECT(sum>scale,sum,scale);
   }
   if(mode==2){
    Q factor=zero+1.;bool low=false;
    for(int z=0;z<8;++z)if(scale[z]<0x1p-450){factor[z]=0x1p512;logs[z]-=512*std::log(2.);low=true;}
    if(low)for(int j=0;j<=m;++j){cm[j]*=factor;ci[j]*=factor;cd[j]*=factor;}
   }else{
    for(int z=0;z<8;++z){if(!(scale[z]>0))scale[z]=1.;logs[z]+=std::log(scale[z]);}
    for(int j=0;j<=m;++j){cm[j]/=scale;ci[j]/=scale;cd[j]/=scale;}
   }
   std::swap(pm,cm);std::swap(pi,ci);std::swap(pd,cd);
   bool finish=false;for(int z=0;z<take;++z)if(rn[offset+z]==i)finish=true;
   if(finish){
    Q sum=zero;if(mode!=1)for(int j=0;j<=m;++j)sum+=pm[j]+pi[j];
    for(int z=0;z<take;++z)if(rn[offset+z]==i){int ml=hn[offset+z];
     double v=mode==1&&ml>0?pm[ml][z]+pi[ml][z]+pd[ml][z]:sum[z];
     output[offset+z]=std::log(std::max(1e-300,v))+logs[z];}
   }
  }
  for(int z=0;z<take;++z)if(rn[offset+z]<=0||hn[offset+z]<=0)output[offset+z]=-INFINITY;
 }
}
#endif

extern "C" int pairhmm_pairs(const char* const* reads,const unsigned char* const* quals,const int* rn,
 const char* const* haps,const int* hn,int count,int mode,int lanes,double* output){
 if(count<0||mode<0||mode>2)return 1;
 #if defined(__x86_64__)
 if((lanes==0||lanes>=8)&&__builtin_cpu_supports("avx512f")){batch_8(reads,quals,rn,haps,hn,count,mode,output);return 0;}
 if((lanes==0||lanes>=4)&&__builtin_cpu_supports("avx2")){batch_4(reads,quals,rn,haps,hn,count,mode,output);return 0;}
 #endif
 batch_2(reads,quals,rn,haps,hn,count,mode,output);return 0;
}
