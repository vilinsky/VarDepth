// Nonnegative PairHMM with explicit accounting for discarded probability mass.
// Each discarded state is an absorbing loss: all its future terminal mass is
// bounded above by its current mass because transitions/emissions are <=1 and
// outgoing transition sums are <=1. This bound is in real arithmetic; callers
// must include a floating-point allowance and retain an exact fallback.
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdint>
extern "C" int sparse_pairs(const char* const* reads,const unsigned char* const* quals,
 const int* rn,const char* const* haps,const int* hn,int count,int mode,
 double threshold,double* likelihood,double* lost,uint64_t* cells){
 if(count<0||mode<0||mode>2||!(threshold>0))return 1;
 static const std::vector<double> errors=[](){std::vector<double> e(256);for(int q=0;q<256;++q)e[q]=std::pow(10.,-std::min(q,60)/10.);return e;}();
 std::vector<double> pm,pi,pd,cm,ci,cd;*cells=0;
 for(int pair=0;pair<count;++pair){
  int n=rn[pair],m=hn[pair];likelihood[pair]=0.;lost[pair]=0.;
  if(n<1||m<1)continue;
  pm.assign(m+1,mode==1?0.:1./(m+1));pi.assign(m+1,0.);pd.assign(m+1,0.);cm.resize(m+1);ci.resize(m+1);cd.resize(m+1);
  int low=0,high=m;
  if(mode==1){pm[0]=1.;high=0;for(int j=1;j<=m;++j){double v=.001*std::pow(.1,j-1);if(v>=threshold){pd[j]=v;high=j;}else lost[pair]+=v;}}
  for(int i=1;i<=n;++i){
   std::fill(cm.begin(),cm.end(),0.);std::fill(ci.begin(),ci.end(),0.);std::fill(cd.begin(),cd.end(),0.);
   ci[0]=.25*(.001*pm[0]+.1*pi[0]);int nextlow=m+1,nexthigh=-1;
   if(ci[0]>=threshold){nextlow=0;nexthigh=0;}else{lost[pair]+=ci[0];ci[0]=0.;}
   double error=errors[quals[pair][i-1]];int last=std::min(m,high+1);
   for(int j=std::max(1,low);j<=m;++j){
    ++*cells;
    double emit=reads[pair][i-1]==haps[pair][j-1]?1-error:error/3.;
    cm[j]=emit*((1-2*.001)*pm[j-1]+(1-.1)*(pi[j-1]+pd[j-1]));
    ci[j]=.25*(.001*pm[j]+.1*pi[j]);cd[j]=.001*cm[j-1]+.1*cd[j-1];
    double mass=cm[j]+ci[j]+cd[j];
    if(mass<threshold){lost[pair]+=mass;cm[j]=ci[j]=cd[j]=0.;if(j>last)break;}
    else{nextlow=std::min(nextlow,j);nexthigh=j;}
   }
   pm.swap(cm);pi.swap(ci);pd.swap(cd);low=nextlow;high=nexthigh;
   if(high<0)break;
  }
  if(high>=0){
   if(mode==1)likelihood[pair]=pm[m]+pi[m]+pd[m];
   else for(int j=0;j<=m;++j)likelihood[pair]+=pm[j]+pi[j];
  }
 }
 return 0;
}
