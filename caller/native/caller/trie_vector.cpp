// Independently implemented prefix sharing across haplotypes, SIMD across reads.
// Unbanded double precision, with the original dense kernel for tiny likelihoods.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <new>
struct Aligned {
 void* p=nullptr;size_t bytes=0;
 ~Aligned(){if(p)::operator delete(p,std::align_val_t(64));}
 void reserve(size_t n){if(n>bytes){if(p)::operator delete(p,std::align_val_t(64));p=::operator new(n,std::align_val_t(64));bytes=n;}}
};
static const double* errors(){static const std::vector<double> e=[](){std::vector<double> x(256);for(int i=0;i<256;++i)x[i]=std::pow(10.,-std::min(i,60)/10.);return x;}();return e.data();}
#define SELECT(mask,a,b) ((Q)(((I)(a)&(mask))|((I)(b)&~(mask))))

 static void compute_2(const char* const* reads,const unsigned char* const* quals,const int* rn,int nr,
 const char* const* haps,const int* hn,int nh,int mode,double* out,uint64_t* cells){
 using Q=double __attribute__((vector_size(16)));using I=long long __attribute__((vector_size(16)));
 thread_local Aligned buffer;
 int suffix=0;
 if(nh>1){
  int minimum=*std::min_element(hn,hn+nh);
  while(suffix<minimum){char c=haps[0][hn[0]-suffix-1];bool same=true;for(int h=1;h<nh;++h)if(haps[h][hn[h]-suffix-1]!=c){same=false;break;}if(!same)break;++suffix;}
  if(suffix<16)suffix=0;
 }
 int depth=0;std::vector<int> shared(nh);
 for(int h=0;h<nh;++h){depth=std::max(depth,hn[h]-suffix);if(h)while(shared[h]<std::min(hn[h],hn[h-1])-suffix&&haps[h][shared[h]]==haps[h-1][shared[h]])++shared[h];}
 // Only fork columns are reused by another haplotype. All other columns
 // can share two rolling slots, with the identical cell arithmetic/order.
 std::vector<int> slots(depth+1,-1);slots[0]=0;int saved=1;
 for(int h=1;h<nh;++h)if(slots[shared[h]]<0)slots[shared[h]]=saved++;
 for(int j=1;j<=depth;++j)if(slots[j]<0)slots[j]=saved+(j&1);
 std::vector<double> boundary(depth+1,0.);
 if(mode==1)for(int j=1;j<=depth;++j)boundary[j]=.001*std::pow(.1,j-1);
 for(int first=0;first<nr;first+=2){
  int take=std::min(2,nr-first),n=0;for(int a=0;a<take;++a)n=std::max(n,rn[first+a]);size_t stride=n+1;
  size_t data_size=size_t(saved+2)*stride*3;
  buffer.reserve((data_size+depth+1+12*stride)*sizeof(Q));
  Q* memory=(Q*)buffer.p;Q* sum=memory+data_size;Q* error=sum+depth+1;Q* mismatch=error+stride;I* base=(I*)(mismatch+stride);
  I read_lengths={};for(int a=0;a<2;++a)read_lengths[a]=a<take?rn[first+a]:-1;
  Q zero={};Q one=zero+1.,quarter=zero+.25,g=zero+.001,x=zero+.1,match=zero+(1-2*.001),exit=zero+(1-.1);
  for(int i=0;i<=n;++i){memory[i]=memory[stride+i]=memory[2*stride+i]=zero;}
  memory[0]=one;
  for(int i=1;i<=n;++i){
   memory[stride+i]=quarter*(g*memory[i-1]+x*memory[stride+i-1]);
   for(int a=0;a<2;++a){bool valid=a<take&&i<=rn[first+a];error[i][a]=valid?errors()[quals[first+a][i-1]]:.001;base[i][a]=valid?(unsigned char)reads[first+a][i-1]:0;}
   mismatch[i]=error[i]/3.;error[i]=one-error[i]; // Same rounded emissions, reused at every haplotype column.
  }
  sum[0]=zero;
  for(int a=0;a<take;++a)sum[0][a]=memory[rn[first+a]][a]+memory[stride+rn[first+a]][a];
  Q free_starts=zero;Q* coefficient=(Q*)(base+stride);
  if(suffix){
   // Backward continuation over a suffix common to every allele. Free-start
   // mass inside the suffix is added separately; all paths remain included.
   Q* right=coefficient+3*stride;Q* current=right+3*stride;
   for(size_t z=0;z<3*stride;++z)right[z]=zero;
   for(int j=suffix-1;j>=0;--j){
    I hc={};if(j+1<suffix)hc+=(unsigned char)haps[0][hn[0]-suffix+j+1];
    for(int i=n;i>=0;--i){
     I row={};row+=i;I terminal=(mode!=1||j==suffix-1)?(I)(read_lengths==row):I{};I active=mode==1?(read_lengths>=row):(read_lengths>row);
     Q diagonal=zero,insert=zero;
     if(i<n){Q emit=SELECT(base[i+1]==hc,error[i+1],mismatch[i+1]);if(j+1<suffix)diagonal=emit*right[i+1];insert=quarter*current[stride+i+1];}
     Q deletion=(i||mode==1)?right[2*stride+i]:zero;
     current[i]=SELECT(terminal,one,SELECT(active,match*diagonal+g*insert+g*deletion,zero));
     current[stride+i]=SELECT(terminal,one,SELECT(active,exit*diagonal+x*insert,zero));
     current[2*stride+i]=(mode==1)?SELECT(terminal,one,SELECT(active,exit*diagonal+x*right[2*stride+i],zero)):SELECT(active,exit*diagonal+x*right[2*stride+i],zero);
    }
    if(mode!=1)free_starts+=current[0];std::swap(current,right);
   }
   I hc={};hc+=(unsigned char)haps[0][hn[0]-suffix];
   for(int i=0;i<=n;++i){
    I row={};row+=i;I active=mode==1?(read_lengths>=row):(read_lengths>row);
    Q diagonal=zero;
    if(i<n){Q emit=SELECT(base[i+1]==hc,error[i+1],mismatch[i+1]);diagonal=emit*right[i+1];}
    coefficient[i]=SELECT(active,match*diagonal+((i||mode==1)?g*right[2*stride+i]:zero),zero);
    coefficient[stride+i]=SELECT(active,exit*diagonal,zero);
    coefficient[2*stride+i]=SELECT(active,exit*diagonal+x*right[2*stride+i],zero);
   }
  }

  for(int h=0;h<nh;++h){
   int length=hn[h]-suffix;
   for(int j=shared[h]+1;j<=length;++j){
    Q* left=memory+size_t(slots[j-1])*stride*3;Q* now=memory+size_t(slots[j])*stride*3;
    Q* lm=left,*li=left+stride,*ld=left+2*stride,*m=now,*ins=now+stride,*del=now+2*stride;
    m[0]=mode==1?zero:one;ins[0]=zero;del[0]=zero+boundary[j];
    I hapcode={};hapcode+=(unsigned char)haps[h][j-1];
    for(int i=1;i<=n;++i){
     Q emit=SELECT(base[i]==hapcode,error[i],mismatch[i]);
     m[i]=emit*(match*lm[i-1]+exit*(li[i-1]+ld[i-1]));
     ins[i]=quarter*(g*m[i-1]+x*ins[i-1]);del[i]=g*lm[i]+x*ld[i];
    }
    *cells+=uint64_t(n)*take;
    if(mode!=1){sum[j]=sum[j-1];for(int a=0;a<take;++a){int z=rn[first+a];sum[j][a]+=m[z][a]+ins[z][a];}}
   }
   Q* last=memory+size_t(slots[length])*stride*3;Q probability=mode==1?zero:sum[length]+free_starts;
   if(suffix)for(int i=0;i<=n;++i)probability+=last[i]*coefficient[i]+last[stride+i]*coefficient[stride+i]+last[2*stride+i]*coefficient[2*stride+i];
   for(int a=0;a<take;++a){int z=rn[first+a];double value=mode==1?(suffix?probability[a]:last[z][a]+last[stride+z][a]+last[2*stride+z][a]):probability[a]/(hn[h]+1);
    out[size_t(first+a)*nh+h]=z==0||hn[h]==0?-INFINITY:value>1e-200?std::log(value):NAN;}
  }
 }
}

#if defined(__x86_64__)

__attribute__((target("avx2"))) static void compute_4(const char* const* reads,const unsigned char* const* quals,const int* rn,int nr,
 const char* const* haps,const int* hn,int nh,int mode,double* out,uint64_t* cells){
 using Q=double __attribute__((vector_size(32)));using I=long long __attribute__((vector_size(32)));
 thread_local Aligned buffer;
 int suffix=0;
 if(nh>1){
  int minimum=*std::min_element(hn,hn+nh);
  while(suffix<minimum){char c=haps[0][hn[0]-suffix-1];bool same=true;for(int h=1;h<nh;++h)if(haps[h][hn[h]-suffix-1]!=c){same=false;break;}if(!same)break;++suffix;}
  if(suffix<16)suffix=0;
 }
 int depth=0;std::vector<int> shared(nh);
 for(int h=0;h<nh;++h){depth=std::max(depth,hn[h]-suffix);if(h)while(shared[h]<std::min(hn[h],hn[h-1])-suffix&&haps[h][shared[h]]==haps[h-1][shared[h]])++shared[h];}
 // Only fork columns are reused by another haplotype. All other columns
 // can share two rolling slots, with the identical cell arithmetic/order.
 std::vector<int> slots(depth+1,-1);slots[0]=0;int saved=1;
 for(int h=1;h<nh;++h)if(slots[shared[h]]<0)slots[shared[h]]=saved++;
 for(int j=1;j<=depth;++j)if(slots[j]<0)slots[j]=saved+(j&1);
 std::vector<double> boundary(depth+1,0.);
 if(mode==1)for(int j=1;j<=depth;++j)boundary[j]=.001*std::pow(.1,j-1);
 for(int first=0;first<nr;first+=4){
  int take=std::min(4,nr-first),n=0;for(int a=0;a<take;++a)n=std::max(n,rn[first+a]);size_t stride=n+1;
  size_t data_size=size_t(saved+2)*stride*3;
  buffer.reserve((data_size+depth+1+12*stride)*sizeof(Q));
  Q* memory=(Q*)buffer.p;Q* sum=memory+data_size;Q* error=sum+depth+1;Q* mismatch=error+stride;I* base=(I*)(mismatch+stride);
  I read_lengths={};for(int a=0;a<4;++a)read_lengths[a]=a<take?rn[first+a]:-1;
  Q zero={};Q one=zero+1.,quarter=zero+.25,g=zero+.001,x=zero+.1,match=zero+(1-2*.001),exit=zero+(1-.1);
  for(int i=0;i<=n;++i){memory[i]=memory[stride+i]=memory[2*stride+i]=zero;}
  memory[0]=one;
  for(int i=1;i<=n;++i){
   memory[stride+i]=quarter*(g*memory[i-1]+x*memory[stride+i-1]);
   for(int a=0;a<4;++a){bool valid=a<take&&i<=rn[first+a];error[i][a]=valid?errors()[quals[first+a][i-1]]:.001;base[i][a]=valid?(unsigned char)reads[first+a][i-1]:0;}
   mismatch[i]=error[i]/3.;error[i]=one-error[i]; // Same rounded emissions, reused at every haplotype column.
  }
  sum[0]=zero;
  for(int a=0;a<take;++a)sum[0][a]=memory[rn[first+a]][a]+memory[stride+rn[first+a]][a];
  Q free_starts=zero;Q* coefficient=(Q*)(base+stride);
  if(suffix){
   // Backward continuation over a suffix common to every allele. Free-start
   // mass inside the suffix is added separately; all paths remain included.
   Q* right=coefficient+3*stride;Q* current=right+3*stride;
   for(size_t z=0;z<3*stride;++z)right[z]=zero;
   for(int j=suffix-1;j>=0;--j){
    I hc={};if(j+1<suffix)hc+=(unsigned char)haps[0][hn[0]-suffix+j+1];
    for(int i=n;i>=0;--i){
     I row={};row+=i;I terminal=(mode!=1||j==suffix-1)?(I)(read_lengths==row):I{};I active=mode==1?(read_lengths>=row):(read_lengths>row);
     Q diagonal=zero,insert=zero;
     if(i<n){Q emit=SELECT(base[i+1]==hc,error[i+1],mismatch[i+1]);if(j+1<suffix)diagonal=emit*right[i+1];insert=quarter*current[stride+i+1];}
     Q deletion=(i||mode==1)?right[2*stride+i]:zero;
     current[i]=SELECT(terminal,one,SELECT(active,match*diagonal+g*insert+g*deletion,zero));
     current[stride+i]=SELECT(terminal,one,SELECT(active,exit*diagonal+x*insert,zero));
     current[2*stride+i]=(mode==1)?SELECT(terminal,one,SELECT(active,exit*diagonal+x*right[2*stride+i],zero)):SELECT(active,exit*diagonal+x*right[2*stride+i],zero);
    }
    if(mode!=1)free_starts+=current[0];std::swap(current,right);
   }
   I hc={};hc+=(unsigned char)haps[0][hn[0]-suffix];
   for(int i=0;i<=n;++i){
    I row={};row+=i;I active=mode==1?(read_lengths>=row):(read_lengths>row);
    Q diagonal=zero;
    if(i<n){Q emit=SELECT(base[i+1]==hc,error[i+1],mismatch[i+1]);diagonal=emit*right[i+1];}
    coefficient[i]=SELECT(active,match*diagonal+((i||mode==1)?g*right[2*stride+i]:zero),zero);
    coefficient[stride+i]=SELECT(active,exit*diagonal,zero);
    coefficient[2*stride+i]=SELECT(active,exit*diagonal+x*right[2*stride+i],zero);
   }
  }

  for(int h=0;h<nh;++h){
   int length=hn[h]-suffix;
   for(int j=shared[h]+1;j<=length;++j){
    Q* left=memory+size_t(slots[j-1])*stride*3;Q* now=memory+size_t(slots[j])*stride*3;
    Q* lm=left,*li=left+stride,*ld=left+2*stride,*m=now,*ins=now+stride,*del=now+2*stride;
    m[0]=mode==1?zero:one;ins[0]=zero;del[0]=zero+boundary[j];
    I hapcode={};hapcode+=(unsigned char)haps[h][j-1];
    for(int i=1;i<=n;++i){
     Q emit=SELECT(base[i]==hapcode,error[i],mismatch[i]);
     m[i]=emit*(match*lm[i-1]+exit*(li[i-1]+ld[i-1]));
     ins[i]=quarter*(g*m[i-1]+x*ins[i-1]);del[i]=g*lm[i]+x*ld[i];
    }
    *cells+=uint64_t(n)*take;
    if(mode!=1){sum[j]=sum[j-1];for(int a=0;a<take;++a){int z=rn[first+a];sum[j][a]+=m[z][a]+ins[z][a];}}
   }
   Q* last=memory+size_t(slots[length])*stride*3;Q probability=mode==1?zero:sum[length]+free_starts;
   if(suffix)for(int i=0;i<=n;++i)probability+=last[i]*coefficient[i]+last[stride+i]*coefficient[stride+i]+last[2*stride+i]*coefficient[2*stride+i];
   for(int a=0;a<take;++a){int z=rn[first+a];double value=mode==1?(suffix?probability[a]:last[z][a]+last[stride+z][a]+last[2*stride+z][a]):probability[a]/(hn[h]+1);
    out[size_t(first+a)*nh+h]=z==0||hn[h]==0?-INFINITY:value>1e-200?std::log(value):NAN;}
  }
 }
}

__attribute__((target("avx512f"))) static void compute_8(const char* const* reads,const unsigned char* const* quals,const int* rn,int nr,
 const char* const* haps,const int* hn,int nh,int mode,double* out,uint64_t* cells){
 using Q=double __attribute__((vector_size(64)));using I=long long __attribute__((vector_size(64)));
 thread_local Aligned buffer;
 int suffix=0;
 if(nh>1){
  int minimum=*std::min_element(hn,hn+nh);
  while(suffix<minimum){char c=haps[0][hn[0]-suffix-1];bool same=true;for(int h=1;h<nh;++h)if(haps[h][hn[h]-suffix-1]!=c){same=false;break;}if(!same)break;++suffix;}
  if(suffix<16)suffix=0;
 }
 int depth=0;std::vector<int> shared(nh);
 for(int h=0;h<nh;++h){depth=std::max(depth,hn[h]-suffix);if(h)while(shared[h]<std::min(hn[h],hn[h-1])-suffix&&haps[h][shared[h]]==haps[h-1][shared[h]])++shared[h];}
 // Only fork columns are reused by another haplotype. All other columns
 // can share two rolling slots, with the identical cell arithmetic/order.
 std::vector<int> slots(depth+1,-1);slots[0]=0;int saved=1;
 for(int h=1;h<nh;++h)if(slots[shared[h]]<0)slots[shared[h]]=saved++;
 for(int j=1;j<=depth;++j)if(slots[j]<0)slots[j]=saved+(j&1);
 std::vector<double> boundary(depth+1,0.);
 if(mode==1)for(int j=1;j<=depth;++j)boundary[j]=.001*std::pow(.1,j-1);
 for(int first=0;first<nr;first+=8){
  int take=std::min(8,nr-first),n=0;for(int a=0;a<take;++a)n=std::max(n,rn[first+a]);size_t stride=n+1;
  size_t data_size=size_t(saved+2)*stride*3;
  buffer.reserve((data_size+depth+1+12*stride)*sizeof(Q));
  Q* memory=(Q*)buffer.p;Q* sum=memory+data_size;Q* error=sum+depth+1;Q* mismatch=error+stride;I* base=(I*)(mismatch+stride);
  I read_lengths={};for(int a=0;a<8;++a)read_lengths[a]=a<take?rn[first+a]:-1;
  Q zero={};Q one=zero+1.,quarter=zero+.25,g=zero+.001,x=zero+.1,match=zero+(1-2*.001),exit=zero+(1-.1);
  for(int i=0;i<=n;++i){memory[i]=memory[stride+i]=memory[2*stride+i]=zero;}
  memory[0]=one;
  for(int i=1;i<=n;++i){
   memory[stride+i]=quarter*(g*memory[i-1]+x*memory[stride+i-1]);
   for(int a=0;a<8;++a){bool valid=a<take&&i<=rn[first+a];error[i][a]=valid?errors()[quals[first+a][i-1]]:.001;base[i][a]=valid?(unsigned char)reads[first+a][i-1]:0;}
   mismatch[i]=error[i]/3.;error[i]=one-error[i]; // Same rounded emissions, reused at every haplotype column.
  }
  sum[0]=zero;
  for(int a=0;a<take;++a)sum[0][a]=memory[rn[first+a]][a]+memory[stride+rn[first+a]][a];
  Q free_starts=zero;Q* coefficient=(Q*)(base+stride);
  if(suffix){
   // Backward continuation over a suffix common to every allele. Free-start
   // mass inside the suffix is added separately; all paths remain included.
   Q* right=coefficient+3*stride;Q* current=right+3*stride;
   for(size_t z=0;z<3*stride;++z)right[z]=zero;
   for(int j=suffix-1;j>=0;--j){
    I hc={};if(j+1<suffix)hc+=(unsigned char)haps[0][hn[0]-suffix+j+1];
    for(int i=n;i>=0;--i){
     I row={};row+=i;I terminal=(mode!=1||j==suffix-1)?(I)(read_lengths==row):I{};I active=mode==1?(read_lengths>=row):(read_lengths>row);
     Q diagonal=zero,insert=zero;
     if(i<n){Q emit=SELECT(base[i+1]==hc,error[i+1],mismatch[i+1]);if(j+1<suffix)diagonal=emit*right[i+1];insert=quarter*current[stride+i+1];}
     Q deletion=(i||mode==1)?right[2*stride+i]:zero;
     current[i]=SELECT(terminal,one,SELECT(active,match*diagonal+g*insert+g*deletion,zero));
     current[stride+i]=SELECT(terminal,one,SELECT(active,exit*diagonal+x*insert,zero));
     current[2*stride+i]=(mode==1)?SELECT(terminal,one,SELECT(active,exit*diagonal+x*right[2*stride+i],zero)):SELECT(active,exit*diagonal+x*right[2*stride+i],zero);
    }
    if(mode!=1)free_starts+=current[0];std::swap(current,right);
   }
   I hc={};hc+=(unsigned char)haps[0][hn[0]-suffix];
   for(int i=0;i<=n;++i){
    I row={};row+=i;I active=mode==1?(read_lengths>=row):(read_lengths>row);
    Q diagonal=zero;
    if(i<n){Q emit=SELECT(base[i+1]==hc,error[i+1],mismatch[i+1]);diagonal=emit*right[i+1];}
    coefficient[i]=SELECT(active,match*diagonal+((i||mode==1)?g*right[2*stride+i]:zero),zero);
    coefficient[stride+i]=SELECT(active,exit*diagonal,zero);
    coefficient[2*stride+i]=SELECT(active,exit*diagonal+x*right[2*stride+i],zero);
   }
  }

  for(int h=0;h<nh;++h){
   int length=hn[h]-suffix;
   for(int j=shared[h]+1;j<=length;++j){
    Q* left=memory+size_t(slots[j-1])*stride*3;Q* now=memory+size_t(slots[j])*stride*3;
    Q* lm=left,*li=left+stride,*ld=left+2*stride,*m=now,*ins=now+stride,*del=now+2*stride;
    m[0]=mode==1?zero:one;ins[0]=zero;del[0]=zero+boundary[j];
    I hapcode={};hapcode+=(unsigned char)haps[h][j-1];
    for(int i=1;i<=n;++i){
     Q emit=SELECT(base[i]==hapcode,error[i],mismatch[i]);
     m[i]=emit*(match*lm[i-1]+exit*(li[i-1]+ld[i-1]));
     ins[i]=quarter*(g*m[i-1]+x*ins[i-1]);del[i]=g*lm[i]+x*ld[i];
    }
    *cells+=uint64_t(n)*take;
    if(mode!=1){sum[j]=sum[j-1];for(int a=0;a<take;++a){int z=rn[first+a];sum[j][a]+=m[z][a]+ins[z][a];}}
   }
   Q* last=memory+size_t(slots[length])*stride*3;Q probability=mode==1?zero:sum[length]+free_starts;
   if(suffix)for(int i=0;i<=n;++i)probability+=last[i]*coefficient[i]+last[stride+i]*coefficient[stride+i]+last[2*stride+i]*coefficient[2*stride+i];
   for(int a=0;a<take;++a){int z=rn[first+a];double value=mode==1?(suffix?probability[a]:last[z][a]+last[stride+z][a]+last[2*stride+z][a]):probability[a]/(hn[h]+1);
    out[size_t(first+a)*nh+h]=z==0||hn[h]==0?-INFINITY:value>1e-200?std::log(value):NAN;}
  }
 }
}

#endif

extern "C" int trie_many(const char* const* reads,const unsigned char* const* quals,const int* rn,int nr,
 const char* const* haps,const int* hn,int nh,int mode,int lanes,double* out,uint64_t* cells){
 if(nr<0||nh<0||mode<0||mode>2)return 1;*cells=0;
 #if defined(__x86_64__)
 if((lanes==0||lanes==8)&&__builtin_cpu_supports("avx512f")){compute_8(reads,quals,rn,nr,haps,hn,nh,mode,out,cells);return 0;}
 if((lanes==0||lanes==4)&&__builtin_cpu_supports("avx2")){compute_4(reads,quals,rn,nr,haps,hn,nh,mode,out,cells);return 0;}
 #endif
 compute_2(reads,quals,rn,nr,haps,hn,nh,mode,out,cells);return 0;
}
