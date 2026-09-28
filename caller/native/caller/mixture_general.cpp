// Finite artifact/balance mixture with exact repeated-observation reuse.
// All observations contribute in their original order; no weighted shortcuts.
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>
#include "vector_log.hpp"
static double add(double x,double y){if(x==y)return x+std::log(2.);return std::max(x,y)+std::log1p(std::exp(-std::abs(x-y)));}
extern "C" void mixture_general(int rows,int n,const double* lh,double prior,int triangular,double* out){
 const double balance[7]={.2,.3,.4,.5,.6,.7,.8},bw[7]={.025,.1,.225,.3,.225,.1,.025};
 const double theta[4]={.02,.1,.2,.3},tw[4]={.4,.35,.2,.05};
 std::vector<int> aa,bb;
 if(triangular){for(int b=0;b<n;++b)for(int a=0;a<=b;++a){aa.push_back(a);bb.push_back(b);}}
 else{for(int a=0;a<n;++a)for(int b=0;b<n;++b){aa.push_back(a);bb.push_back(b);}}
 int pairs=aa.size(),mixes=prior<0?1:7+(prior>0?4*n:0);
 std::vector<double> totals(size_t(pairs)*mixes,0.),scratch(totals.size());
 std::map<std::vector<double>,std::vector<double>> cache;
 size_t capacity=std::min(size_t(128),size_t(16*1024*1024)/std::max(size_t(1),totals.size()*8));
 for(int row=0;row<rows;++row){
  const double* v=lh+size_t(row)*n;std::vector<double> key(v,v+n);auto it=cache.find(key);const double* values;
  if(it==cache.end()){
   for(int g=0;g<pairs;++g){double a=v[aa[g]],b=v[bb[g]];double* dest=scratch.data()+size_t(g)*mixes;
    if(prior<0)dest[0]=std::max(1e-300,(a+b)/2);
    else{
     for(int j=0;j<7;++j)dest[j]=std::max(1e-300,balance[j]*a+(1-balance[j])*b);
     if(prior>0)for(int j=0;j<4;++j)for(int c=0;c<n;++c)dest[7+j*n+c]=std::max(1e-300,(1-theta[j])*(a+b)/2+theta[j]*v[c]);
    }
   }
   vector_log(scratch.data(),scratch.size());
   if(cache.size()<capacity)values=cache.emplace(std::move(key),scratch).first->second.data();
   else values=scratch.data();
  }else values=it->second.data();
  for(size_t j=0;j<totals.size();++j)totals[j]+=values[j];
 }
 for(int g=0;g<pairs;++g){const double* v=totals.data()+size_t(g)*mixes;
  if(prior<0){out[g]=v[0];continue;}
  double clean=v[0]+std::log(bw[0]);for(int j=1;j<7;++j)clean=add(clean,v[j]+std::log(bw[j]));
  if(prior>0){double art=v[7]+std::log(tw[0]/n);for(int j=1;j<4*n;++j)art=add(art,v[7+j]+std::log(tw[j/n]/n));out[g]=add(std::log1p(-prior)+clean,std::log(prior)+art);}
  else out[g]=clean;
 }
}
