// Ordered summation with exact reuse of binary donor likelihood values.
// Logs are computed by NumPy once per row and allele, retaining its arithmetic.
#include <cstdint>
extern "C" void binary_sums(int m,int k,const double* h,const double* table,double* out){
 for(int a=0;a<k;++a)out[a]=0.;
 for(int t=0;t<m;++t)for(int a=0;a<k;++a)out[a]+=table[2*t+(h[(uint64_t)t*k+a]==1.)];
}

// Binary panels occupy one byte per donor. The reduction for each donor still
// visits markers in their original order; only independent columns vectorize.
extern "C" void binary_sums_u8(int m,int k,const uint8_t* h,const double* table,double* out){
 for(int a=0;a<k;++a)out[a]=0.;
 for(int t=0;t<m;++t){
  const uint8_t* row=h+(uint64_t)t*k;
  double ref=table[2*t],alt=table[2*t+1];
  for(int a=0;a<k;++a)out[a]+=row[a]?alt:ref;
 }
}
