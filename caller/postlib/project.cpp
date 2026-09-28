// Independently implemented minimum-edit alignment used only to map candidate
// sequence coordinates. Read likelihoods are evaluated by the PairHMM.
#include <algorithm>
#include <cstdint>
#include <vector>
extern "C" int candidate_projection(const char* ref,int n,const char* alt,int m,char* ops) {
 if(n<1 || m<1 || n>4096 || m>4096)return -1;
 std::vector<uint16_t> dp(size_t(n+1)*(m+1));
 auto at=[&](int i,int j)->uint16_t& {return dp[size_t(i)*(m+1)+j];};
 for(int i=0;i<=n;++i)at(i,0)=i;for(int j=0;j<=m;++j)at(0,j)=j;
 for(int i=1;i<=n;++i)for(int j=1;j<=m;++j)
  at(i,j)=std::min({int(at(i-1,j-1))+(ref[i-1]!=alt[j-1]),int(at(i-1,j))+1,int(at(i,j-1))+1});
 int i=n,j=m,t=0;
 while(i||j) {
  if(i&&j&&at(i,j)==at(i-1,j-1)+(ref[i-1]!=alt[j-1])){ops[t++]='M';--i;--j;}
  else if(i&&at(i,j)==at(i-1,j)+1){ops[t++]='D';--i;}
  else {ops[t++]='I';--j;}
 }
 std::reverse(ops,ops+t);return t;
}
