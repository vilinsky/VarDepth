// Biallelic SNV expert features, with the same diploid priors and FP64
// arithmetic. Entropy is evaluated by the established NumPy float32 path.
#include <algorithm>
#include <cmath>
#include <cstdint>
static bool distribution(const double* v,float* p){
 double sum=(v[0]+v[1])+v[2];if(!(sum>0))return false;
 for(int i=0;i<3;++i){if(v[i]<0||!std::isfinite(v[i]))return false;p[i]=v[i]/sum;}return true;
}
extern "C" int snv_bundle(const double* source,int flags,double dp,double cs,double af,
 double ad0,double ad1,int reads_only,float* ps,uint8_t* available,float* features,float* genotype){
 std::fill(ps,ps+30,0.f);std::fill(available,available+10,0);std::fill(features,features+420,0.f);std::fill(genotype,genotype+60,0.f);
 int ids[4]={0,1,2,7};
 for(int s=0;s<4;++s)if(flags&(1<<s)){if(!distribution(source+3*s,ps+3*ids[s]))return 1;available[ids[s]]=1;}
 double read_lp[3]={-60.,-60.,-60.};int read_best=0;
 if(flags&16){
  af=std::max(.001,af);double rf=std::max(.001,1-af),total=rf+af;rf/=total;af/=total;
  double prior[3]={rf*rf,(rf*af)*2,af*af},lp[3],posterior[3],weak[3];
  for(int i=0;i<3;++i){lp[i]=(-source[12+i]*std::log(10.))/10.;genotype[20*i+18]=1;genotype[20*i+19]=std::min(10.,source[12+i]/60.);}
  double top=*std::max_element(lp,lp+3);
  for(int i=0;i<3;++i){read_lp[i]=lp[i]-top;posterior[i]=lp[i]+std::log(prior[i]);weak[i]=lp[i]+.25*std::log(prior[i]);}
  top=*std::max_element(posterior,posterior+3);double wtop=*std::max_element(weak,weak+3),z=0,wz=0;
  for(int i=0;i<3;++i){posterior[i]=std::exp(posterior[i]-top);weak[i]=std::exp(weak[i]-wtop);z+=posterior[i];wz+=weak[i];}
  for(int i=0;i<3;++i){posterior[i]/=z;ps[12+i]=posterior[i];if(dp>0)ps[18+i]=weak[i]/wz;}
  read_best=int(std::max_element(posterior,posterior+3)-posterior);available[4]=1;if(dp>0)available[6]=1;
 }
 if(dp<0||ad0<0||ad1<0)return 2;
 double adsum=ad0+ad1,context[14]={std::log1p(dp)/std::log(101.),cs,0.,std::log1p(1.)/std::log(101.),std::log(2.)/std::log(24.),
  std::min(ad0,ad1)/std::max(1.,adsum),std::min(1.,adsum/std::max(1.,dp)),0.,0.,0.,0.,0.,0.,double(reads_only)};
 int winners[10],counts[3]={},n_available=0;
 for(int e=0;e<10;++e){winners[e]=int(std::max_element(ps+3*e,ps+3*e+3)-(ps+3*e));if(available[e]){++counts[winners[e]];++n_available;}}
 for(int e=0;e<10;++e){
  float* f=features+42*e;for(int j=0;j<14;++j)f[j]=context[j];
  float sorted[3]={ps[3*e],ps[3*e+1],ps[3*e+2]};std::sort(sorted,sorted+3);int winner=winners[e];
  f[14]=-std::log(std::max(1e-9,1-double(sorted[2])))/20.;f[15]=0.;f[16]=ps[3*e];f[17]=sorted[2]-sorted[1];
  f[18]=double(counts[winner])/std::max(1,n_available);f[19]=winner==winners[0];f[20]=winner==read_best;f[21]=read_lp[winner]/60.;f[22]=winner!=0;
  f[29]=winner==1;f[32+e]=1.;
 }
 genotype[20]=1.;return 0;
}
