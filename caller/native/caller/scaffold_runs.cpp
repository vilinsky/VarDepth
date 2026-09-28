// Numerical fallback for the unchanged diploid SNP copying model.
// Exactly zero-refresh runs have one donor; sum their emissions in log space.
// Positive refreshes, soft panel alleles, Gibbs order and mutation are retained.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

extern "C" int scaffold_gibbs(int m,int k,const double* h,const double* gl,
 const double* refresh,double mutation,int burn,int retained,int chains,
 uint64_t seed,double* gp){
 if(m<1||k<1||burn<0||retained<1||chains<1||mutation<0||mutation>=.5)return 1;
 std::fill(gp,gp+size_t(m)*3,0.);
 std::vector<int> begin{0},mono(m),neutral(m);
 for(int t=1;t<m;++t)if(refresh[t]>0.)begin.push_back(t);
 begin.push_back(m);const int blocks=int(begin.size())-1;
 std::vector<double> l(size_t(m)*3),logtable(size_t(m)*6);
 std::vector<uint8_t> codes(size_t(m)*k);
 for(int t=0;t<m;++t){
  if(!std::isfinite(refresh[t])||refresh[t]<0.||refresh[t]>1.)return 1;
  double top=std::max({gl[3*t],gl[3*t+1],gl[3*t+2]});
  for(int g=0;g<3;++g){if(!std::isfinite(gl[3*t+g]))return 1;l[3*t+g]=std::exp(std::max(-690.,gl[3*t+g]-top));}
  neutral[t]=l[3*t]==l[3*t+1]&&l[3*t]==l[3*t+2];mono[t]=1;
  for(int a=0;a<k;++a){double v=h[size_t(t)*k+a];if(!std::isfinite(v)||v<0.||v>1.)return 1;
   if(v!=h[size_t(t)*k])mono[t]=0;
   codes[size_t(t)*k+a]=v==0.?0:v==.5?1:v==1.?2:3;
  }
  for(int other=0;other<2;++other)for(int c=0;c<3;++c){double p=mutation+(1-2*mutation)*(.5*c);
   logtable[size_t(t)*6+other*3+c]=std::log((1-p)*l[3*t+other]+p*l[3*t+other+1]);}
 }
 auto prob=[&](int t,int a){return mutation+(1-2*mutation)*h[size_t(t)*k+a];};
 std::vector<double> emissions(size_t(blocks)*k),forward(size_t(blocks)*k),back(k),conditional(k),posterior(k);
 std::mt19937_64 rng(seed);std::uniform_real_distribution<double> uniform(0.,1.);
 auto draw=[&](const double* p){double z=0.;for(int a=0;a<k;++a)z+=p[a];double u=uniform(rng)*z;
  for(int a=0;a<k-1;++a){u-=p[a];if(u<=0.)return a;}return k-1;};
 for(int chain=0;chain<chains;++chain){
  std::vector<int> allele[2]={std::vector<int>(m),std::vector<int>(m)};
  for(int c=0;c<2;++c){int donor=(2*chain+c)%k;
   for(int t=0;t<m;++t)allele[c][t]=uniform(rng)<prob(t,donor);}
  for(int iteration=0;iteration<burn+retained;++iteration)for(int chromosome=0;chromosome<2;++chromosome){
   const auto& other=allele[1-chromosome];std::fill(emissions.begin(),emissions.end(),0.);
   for(int b=0;b<blocks;++b){double* e=emissions.data()+size_t(b)*k;
    for(int t=begin[b];t<begin[b+1];++t){
     if(mono[t]||neutral[t])continue; // donor-independent factor cancels
     const uint8_t* code=codes.data()+size_t(t)*k;
     const double* table=logtable.data()+size_t(t)*6+other[t]*3;
     double l0=l[3*t+other[t]],l1=l[3*t+other[t]+1];
     for(int a=0;a<k;++a){if(code[a]<3)e[a]+=table[code[a]];
      else{double p=prob(t,a);e[a]+=std::log((1-p)*l0+p*l1);}}
    }
    double top=*std::max_element(e,e+k);
    for(int a=0;a<k;++a)e[a]=std::exp(e[a]-top);
    double* f=forward.data()+size_t(b)*k;double r=refresh[begin[b]],z=0.;
    if(b){const double* previous=f-k;for(int a=0;a<k;++a){f[a]=((1-r)*previous[a]+r/k)*e[a];z+=f[a];}}
    else for(int a=0;a<k;++a){f[a]=e[a]/k;z+=f[a];}
    if(!(z>0.)||!std::isfinite(z))return 2;
    for(int a=0;a<k;++a)f[a]/=z;
   }
   if(iteration>=burn){std::fill(back.begin(),back.end(),1.);
    for(int b=blocks-1;b>=0;--b){const double* f=forward.data()+size_t(b)*k;double z=0.;
     for(int a=0;a<k;++a){posterior[a]=f[a]*back[a];z+=posterior[a];}
     if(!(z>0.)||!std::isfinite(z))return 4;
     for(int a=0;a<k;++a)posterior[a]/=z;
     for(int t=begin[b+1]-1;t>=begin[b];--t){double l0=l[3*t+other[t]],l1=l[3*t+other[t]+1],p0=0.,p1=0.;
      if(mono[t]){double p=prob(t,0),v0=(1-p)*l0,v1=p*l1;p0=v0/(v0+v1);p1=v1/(v0+v1);}
      else{double buckets[3]={},table0[3],table1[3];const uint8_t* code=codes.data()+size_t(t)*k;
       for(int c=0;c<3;++c){double p=mutation+(1-2*mutation)*(.5*c),v0=(1-p)*l0,v1=p*l1;table0[c]=v0/(v0+v1);table1[c]=v1/(v0+v1);}
       for(int a=0;a<k;++a){if(code[a]<3)buckets[code[a]]+=posterior[a];
        else{double p=prob(t,a),v0=(1-p)*l0,v1=p*l1;p0+=posterior[a]*v0/(v0+v1);p1+=posterior[a]*v1/(v0+v1);}}
       for(int c=0;c<3;++c){p0+=buckets[c]*table0[c];p1+=buckets[c]*table1[c];}
      }
      gp[3*t+other[t]]+=p0;gp[3*t+other[t]+1]+=p1;
     }
     if(b){const double* e=emissions.data()+size_t(b)*k;double mean=0.,top=0.,r=refresh[begin[b]];
      for(int a=0;a<k;++a){conditional[a]=back[a]*e[a];mean+=conditional[a]/k;}
      for(int a=0;a<k;++a){back[a]=(1-r)*conditional[a]+r*mean;top=std::max(top,back[a]);}
      if(!(top>0.)||!std::isfinite(top))return 4;
      for(int a=0;a<k;++a)back[a]/=top;
     }
    }
   }
   int block=blocks-1,donor=draw(forward.data()+size_t(block)*k);
   for(int t=m-1;t>=0;--t){double p=prob(t,donor),v0=(1-p)*l[3*t+other[t]],v1=p*l[3*t+other[t]+1];
    allele[chromosome][t]=uniform(rng)<v1/(v0+v1);
    if(t){double u=uniform(rng); // consume each original per-site RNG draw
     if(t==begin[block]){const double* previous=forward.data()+size_t(block-1)*k;
      double stay=(1-refresh[t])*previous[donor],change=refresh[t]/k;
      if(u*(stay+change)>=stay)donor=draw(previous);--block;}
     // Zero refresh cannot change donor, including an underflowed prefix.
    }
   }
  }
 }
 double expected=double(chains)*retained*2;
 for(int t=0;t<m;++t){double z=gp[3*t]+gp[3*t+1]+gp[3*t+2];
  if(!(z>0.)||!std::isfinite(z)||std::abs(z-expected)>expected*1e-5)return 3;
  for(int g=0;g<3;++g)gp[3*t+g]/=z;
 }
 return 0;
}
