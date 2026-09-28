// Independent recurrence; float64 default, optional guarded float32 workspace.
// No pruning or fast-math. Log scales and final sums remain float64.
#ifdef VARDEPTH_MIXED
using Real=float;
#else
using Real=double;
#endif
extern "C" __global__ void pairhmm(
    const unsigned char* reads, const unsigned char* haps,
    const double* errors, const int* nr, const int* nh,
    const int count, const int columns, const int anchored,
    Real* workspace, double* result) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= count) return;
  const int n=nr[p], m=nh[p];
  if(n<=0 || m<=0) {result[p]=__longlong_as_double(0xfff0000000000000ULL); return;}
  const long stride=(long)(columns+1)*count;
  Real* pm=workspace+p; Real* pi=pm+stride; Real* pd=pi+stride;
  for(int j=0;j<=m;++j) {
    const long k=(long)j*count;
    pm[k]=anchored?(j==0?1.:0.):1./(m+1);
    pi[k]=0.; pd[k]=anchored&&j>0?.001*pow(.1,double(j-1)):0.;
  }
  double logs=0.;
  for(int i=0;i<n;++i) {
    Real diagonal_m=pm[0],diagonal_i=pi[0],diagonal_d=pd[0];
    pm[0]=0.;pd[0]=0.;pi[0]=Real(.25)*(Real(.001)*diagonal_m+Real(.1)*diagonal_i);
    Real current_m=0.,current_d=0.;
    Real scale=pi[0]; const Real e=errors[(long)i*count+p];
    const unsigned char base=reads[(long)i*count+p];
    for(int j=1;j<=m;++j) {
      const long k=(long)j*count, prev=k-count;
      const Real emit=base==haps[prev+p]?Real(1)-e:e/Real(3);
      const Real old_m=pm[k],old_i=pi[k],old_d=pd[k];
      const Real next_m=emit*(Real(.998)*diagonal_m+Real(.9)*(diagonal_i+diagonal_d));
      const Real next_i=Real(.25)*(Real(.001)*old_m+Real(.1)*old_i);
      const Real next_d=Real(.001)*current_m+Real(.1)*current_d;
      pm[k]=next_m;pi[k]=next_i;pd[k]=next_d;
      diagonal_m=old_m;diagonal_i=old_i;diagonal_d=old_d;
      current_m=next_m;current_d=next_d;
      scale=scale>next_m+next_i+next_d ? scale : next_m+next_i+next_d;
    }
    if(!(scale>0.)) {result[p]=__longlong_as_double(0xfff0000000000000ULL); return;}
    if(anchored) {
      for(int j=0;j<=m;++j) {
        const long k=(long)j*count; pm[k]/=scale; pi[k]/=scale; pd[k]/=scale;
      }
      logs+=log(double(scale));
    }
#ifdef VARDEPTH_MIXED
    else if(scale<0x1p-40f) {
      for(int j=0;j<=m;++j) {
        const long k=(long)j*count; pm[k]*=0x1p64f; pi[k]*=0x1p64f; pd[k]*=0x1p64f;
      }
      logs-=64.*log(2.);
    }
#else
    else if(scale<0x1p-450) {
      for(int j=0;j<=m;++j) {
        const long k=(long)j*count; pm[k]*=0x1p512; pi[k]*=0x1p512; pd[k]*=0x1p512;
      }
      logs-=512.*log(2.);
    }
#endif
  }
  double z=0.;
  if(anchored) {const long k=(long)m*count;z=pm[k]+pi[k]+pd[k];}
  else for(int j=0;j<=m;++j) {const long k=(long)j*count; z+=pm[k]+pi[k];}
#ifdef VARDEPTH_MIXED
  // Tiny terminal mass signals unreliable float32 tails: recompute in float64.
  if(z<0x1p-40) {result[p]=__longlong_as_double(0x7ff8000000000000ULL);return;}
#endif
  result[p]=log(fmax(1e-300,z))+logs;
}
