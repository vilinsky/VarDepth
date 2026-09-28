// Exact two-allele artifact/balance mixture; original observation order.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <array>
#include <utility>
static double add(double x,double y){if(x==y)return x+std::log(2.);return std::max(x,y)+std::log1p(std::exp(-std::abs(x-y)));}
extern "C" void mixture_two(int n,const double* lh,double prior,double* out){
 const double balance[7]={.2,.3,.4,.5,.6,.7,.8},bw[7]={.025,.1,.225,.3,.225,.1,.025};
 const double theta[4]={.02,.1,.2,.3},tw[4]={.4,.35,.2,.05};
 double totals[4][15]={};
 std::map<std::pair<double,double>,std::array<double,60>> cached;
 for(int row=0;row<n;++row){double p=lh[2*row],q=lh[2*row+1];auto key=std::make_pair(p,q);auto it=cached.find(key);
  if(it==cached.end()){
   std::array<double,60> values;double v[2]={p,q};
   for(int a=0;a<2;++a)for(int b=0;b<2;++b){int g=a*2+b;
    for(int j=0;j<7;++j)values[g*15+j]=std::log(std::max(1e-300,balance[j]*v[a]+(1-balance[j])*v[b]));
    for(int j=0;j<4;++j)for(int c=0;c<2;++c)values[g*15+7+2*j+c]=std::log(std::max(1e-300,(1-theta[j])*(v[a]+v[b])/2+theta[j]*v[c]));
   }
   it=cached.emplace(key,values).first;
  }
  for(int g=0;g<4;++g)for(int j=0;j<15;++j)totals[g][j]+=it->second[g*15+j];
 }
 for(int g=0;g<4;++g){double clean=totals[g][0]+std::log(bw[0]);for(int j=1;j<7;++j)clean=add(clean,totals[g][j]+std::log(bw[j]));
  if(prior>0){double art=totals[g][7]+std::log(tw[0]/2);for(int j=1;j<8;++j)art=add(art,totals[g][7+j]+std::log(tw[j/2]/2));out[g]=add(std::log1p(-prior)+clean,std::log(prior)+art);}
  else out[g]=clean;
 }
}
