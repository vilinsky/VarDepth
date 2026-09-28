// Prefix sharing for the same unbanded, double-precision PairHMM model.
// Columns are evaluated once per distinct haplotype prefix. No path pruning.
// Initial semiglobal mass is 1 instead of 1/(m+1); undo that at the terminal.
// Very small values are delegated to the existing normalized dense kernel.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
extern "C" int trie_pairs(const char* const* reads,const unsigned char* const* quals,
 const int* rn,const char* const* haps,const int* hn,const int* groups,
 int group_count,int count,int mode,double* out,uint64_t* cells){
 if(group_count<0||count<0||mode<0||mode>2)return 1;
 static const std::vector<double> err=[](){std::vector<double> v(256);for(int i=0;i<256;++i)v[i]=std::pow(10.,-std::min(i,60)/10.);return v;}();
 *cells=0;
 for(int group=0;group<group_count;++group){
  int begin=groups[group],end=groups[group+1],n=rn[begin],depth=0;
  for(int j=begin;j<end;++j)depth=std::max(depth,hn[j]);
  if(n==0){for(int j=begin;j<end;++j)out[j]=-INFINITY;continue;}
  size_t stride=n+1;
  std::vector<double> memory(size_t(depth+1)*stride*3,0.),sum(depth+1),error(n+1);
  auto column=[&](int j){return memory.data()+size_t(j)*stride*3;};
  double* root=column(0);root[0]=1.;
  for(int i=1;i<=n;++i){root[stride+i]=.25*(.001*root[i-1]+.1*root[stride+i-1]);error[i]=err[quals[begin][i-1]];}
  sum[0]=root[n]+root[stride+n];
  const char* previous=nullptr;int previous_length=0;
  for(int job=begin;job<end;++job){
   int length=hn[job],common=0;
   if(length==0){out[job]=-INFINITY;continue;}
   if(previous)while(common<std::min(previous_length,length)&&previous[common]==haps[job][common])++common;
   for(int j=common+1;j<=length;++j){
    const double* left=column(j-1);const double* lm=left,*li=left+stride,*ld=left+2*stride;
    double* now=column(j),*m=now,*ins=now+stride,*del=now+2*stride;
    m[0]=mode==1?0.:1.;ins[0]=0.;del[0]=mode==1?.001*std::pow(.1,j-1):0.;
    for(int i=1;i<=n;++i){
     double emit=reads[begin][i-1]==haps[job][j-1]?1-error[i]:error[i]/3.;
     m[i]=emit*((1-2*.001)*lm[i-1]+(1-.1)*(li[i-1]+ld[i-1]));
     ins[i]=.25*(.001*m[i-1]+.1*ins[i-1]);del[i]=.001*lm[i]+.1*ld[i];
    }
    *cells+=n;sum[j]=sum[j-1]+m[n]+ins[n];
   }
   double* last=column(length);
   double value=mode==1?last[n]+last[stride+n]+last[2*stride+n]:sum[length]/(length+1);
   // NaN explicitly requests normalization-safe fallback, not a genotype.
   out[job]=value>1e-200?std::log(value):NAN;
   previous=haps[job];previous_length=length;
  }
 }
 return 0;
}
