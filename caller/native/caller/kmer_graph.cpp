// Exact canonical k-mer support from distinct molecules, for k <= 63.
// Fixed-width rolling keys avoid allocating a Python string per read window.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using Word=unsigned __int128;
struct Hash{size_t operator()(Word x) const {return uint64_t(x)^(uint64_t(x>>64)*0x9e3779b97f4a7c15ULL);}};
struct Edge{Word word;int count,novel;};
struct Result{std::vector<Edge> edges;int observed,molecules,k;};
static int code(char c){switch(c){case 'A':return 0;case 'C':return 1;case 'G':return 2;case 'T':return 3;default:return -1;}}
extern "C" void* kmer_build(const char* ref,int nr,const char* const* seq,const int* sizes,const int* molecule,int n,int nm,int k,int minimum){
 if(k<3||k>63||nr<2*k||n<0||nm<0)return nullptr;
 Result* result=new Result;result->k=k;result->molecules=0;
 std::vector<std::unordered_set<Word,Hash>> molecules(nm);
 Word mask=(Word(1)<<(2*k))-1;
 for(int j=0;j<n;++j){Word value=0;int run=0;
  for(int i=0;i<sizes[j];++i){int x=code(seq[j][i]);if(x<0){run=0;value=0;continue;}value=((value<<2)|x)&mask;if(++run>=k)molecules[molecule[j]].insert(value);}
 }
 std::unordered_map<Word,int,Hash> support;
 for(auto& words:molecules){if(!words.empty())++result->molecules;for(auto word:words)++support[word];}
 result->observed=int(support.size());
 std::unordered_set<Word,Hash> reference;Word value=0;
 for(int i=0;i<nr;++i){int x=code(ref[i]);if(x<0){delete result;return nullptr;}value=((value<<2)|x)&mask;if(i+1>=k)reference.insert(value);}
 for(auto word:reference){auto it=support.find(word);result->edges.push_back({word,it==support.end()?0:it->second,0});}
 for(auto [word,count]:support)if(count>=minimum&&!reference.count(word))result->edges.push_back({word,count,1});
 std::sort(result->edges.begin(),result->edges.end(),[](const Edge& a,const Edge& b){return a.word<b.word;});
 return result;
}
extern "C" int kmer_size(void* handle,int* observed,int* molecules){auto* r=(Result*)handle;*observed=r->observed;*molecules=r->molecules;return int(r->edges.size());}
extern "C" void kmer_copy(void* handle,char* words,int* counts,int* novel){auto* r=(Result*)handle;const char* dna="ACGT";
 for(size_t i=0;i<r->edges.size();++i){auto e=r->edges[i];counts[i]=e.count;novel[i]=e.novel;for(int j=r->k-1;j>=0;--j){words[i*r->k+j]=dna[int(e.word&3)];e.word>>=2;}}
}
extern "C" void kmer_free(void* handle){delete (Result*)handle;}
