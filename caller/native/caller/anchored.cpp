// Independent global PairHMM for read substrings with observed flanking anchors.
#include <algorithm>
#include <cmath>
#include <vector>
extern "C" double anchored_pairhmm(const char* read,const unsigned char* qual,int n,
 const char* hap,int m,double gap,double extension){
 if(n<=0||m<=0)return -INFINITY;
 std::vector<double> pm(m+1,0),pi(m+1,0),pd(m+1,0),cm(m+1),ci(m+1),cd(m+1);
 pm[0]=1.;for(int j=1;j<=m;++j)pd[j]=gap*std::pow(extension,j-1);
 double logscale=0;
 for(int i=1;i<=n;++i){
  std::fill(cm.begin(),cm.end(),0);std::fill(ci.begin(),ci.end(),0);std::fill(cd.begin(),cd.end(),0);
  ci[0]=.25*(gap*pm[0]+extension*pi[0]);double error=std::pow(10.,-std::min(60,int(qual[i-1]))/10.);
  for(int j=1;j<=m;++j){
   double emit=read[i-1]==hap[j-1]?1-error:error/3.;
   cm[j]=emit*((1-2*gap)*pm[j-1]+(1-extension)*(pi[j-1]+pd[j-1]));
   ci[j]=.25*(gap*pm[j]+extension*pi[j]);cd[j]=gap*cm[j-1]+extension*cd[j-1];
  }
  double scale=0;for(int j=0;j<=m;++j)scale=std::max(scale,cm[j]+ci[j]+cd[j]);
  if(scale<=0)return -INFINITY;
  for(int j=0;j<=m;++j){cm[j]/=scale;ci[j]/=scale;cd[j]/=scale;}
  logscale+=std::log(scale);pm.swap(cm);pi.swap(ci);pd.swap(cd);
 }
 return std::log(std::max(1e-300,pm[m]+pi[m]+pd[m]))+logscale;
}
