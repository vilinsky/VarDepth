// Experimental independent row-parallel recurrence. One warp per read/haplotype.
// The deletion recurrence is an associative affine scan; no path pruning.
#ifdef VARDEPTH_MIXED
using Real=float;
#else
using Real=double;
#endif
extern "C" __global__ void pairhmm(
    const unsigned char* reads,const unsigned char* haps,const double* errors,
    const int* nr,const int* nh,const int count,const int columns,
    const int read_columns,const int anchored,Real* workspace,double* result) {
  const int p=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
  if(p>=count)return;
  const int n=nr[p],m=nh[p];
  const unsigned mask=0xffffffffu;
  const Real carry_decay=Real(pow(.1,double(lane+1)));
  if(n<=0||m<=0){if(lane==0)result[p]=__longlong_as_double(0xfff0000000000000ULL);return;}
  const long stride=(long)(columns+1)*count;
  Real* pm=workspace+(long)p*(columns+1);Real* pi=pm+stride;Real* pd=pi+stride;
  for(int j=lane;j<=m;j+=32){pm[j]=anchored?(j==0?1.:0.):1./(m+1);pi[j]=0.;pd[j]=anchored&&j>0?.001*pow(.1,double(j-1)):0.;}
  __syncwarp(mask);double logs=0.;
  for(int i=0;i<n;++i){
    Real dm=pm[0],di=pi[0],dd=pd[0];
    Real edge_i=Real(.25)*(Real(.001)*dm+Real(.1)*di);
    __syncwarp(mask);
    if(lane==0){pm[0]=0.;pi[0]=edge_i;pd[0]=0.;}
    Real current_m=0.,current_d=0.,scale=lane==0?edge_i:Real(0);
    const Real e=errors[(long)p*read_columns+i];const unsigned char base=reads[(long)p*read_columns+i];
    for(int begin=1;begin<=m;begin+=32){
      int j=begin+lane;bool valid=j<=m;
      Real om=valid?pm[j]:Real(0),oi=valid?pi[j]:Real(0),od=valid?pd[j]:Real(0);
      Real diagonal_m=__shfl_up_sync(mask,om,1),diagonal_i=__shfl_up_sync(mask,oi,1),diagonal_d=__shfl_up_sync(mask,od,1);
      if(lane==0){diagonal_m=dm;diagonal_i=di;diagonal_d=dd;}
      Real emit=valid&&base==haps[(long)p*columns+j-1]?Real(1)-e:e/Real(3);
      Real nm=emit*(Real(.998)*diagonal_m+Real(.9)*(diagonal_i+diagonal_d));
      Real ni=Real(.25)*(Real(.001)*om+Real(.1)*oi);
      Real left_m=__shfl_up_sync(mask,nm,1);if(lane==0)left_m=current_m;
      Real nd=Real(.001)*left_m,coefficient=Real(.1);
      for(int offset=1;offset<32;offset*=2){Real left=__shfl_up_sync(mask,nd,offset);if(lane>=offset)nd+=coefficient*left;coefficient*=coefficient;}
      nd+=carry_decay*current_d;
      const int last=(m-begin<31?m-begin:31);
      dm=__shfl_sync(mask,om,last);di=__shfl_sync(mask,oi,last);dd=__shfl_sync(mask,od,last);
      current_m=__shfl_sync(mask,nm,last);current_d=__shfl_sync(mask,nd,last);
      if(valid){pm[j]=nm;pi[j]=ni;pd[j]=nd;Real mass=nm+ni+nd;scale=scale>mass?scale:mass;}
      __syncwarp(mask);
    }
    for(int offset=16;offset;offset/=2){Real other=__shfl_down_sync(mask,scale,offset);scale=scale>other?scale:other;}
    scale=__shfl_sync(mask,scale,0);
    if(!(scale>0)){if(lane==0)result[p]=__longlong_as_double(0xfff0000000000000ULL);return;}
    if(anchored){for(int j=lane;j<=m;j+=32){pm[j]/=scale;pi[j]/=scale;pd[j]/=scale;}logs+=log(double(scale));}
#ifdef VARDEPTH_MIXED
    else if(scale<0x1p-40f){for(int j=lane;j<=m;j+=32){pm[j]*=0x1p64f;pi[j]*=0x1p64f;pd[j]*=0x1p64f;}logs-=64.*log(2.);}
#else
    else if(scale<0x1p-450){for(int j=lane;j<=m;j+=32){pm[j]*=0x1p512;pi[j]*=0x1p512;pd[j]*=0x1p512;}logs-=512.*log(2.);}
#endif
    __syncwarp(mask);
  }
  if(lane==0){
    double z=0.;if(anchored)z=pm[m]+pi[m]+pd[m];else for(int j=0;j<=m;++j)z+=pm[j]+pi[j];
#ifdef VARDEPTH_MIXED
    if(z<0x1p-40){result[p]=__longlong_as_double(0x7ff8000000000000ULL);return;}
#endif
    result[p]=log(fmax(1e-300,z))+logs;
  }
}
