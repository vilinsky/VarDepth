// Bounded, invocation-local memoization of deterministic conditional HMMs.
// Keys contain every informative nonconstant donor marker. Sampling is NEVER
// cached: each Gibbs update consumes its original random draws in original order.
#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
struct HMMCacheEntry {
 std::vector<double> fw, scale, posterior;
 bool posterior_ready=false;
 uint64_t used=0;
};
class HMMCache {
 size_t states_, markers_, posterior_, capacity_;
 uint64_t clock_=0;
 std::map<std::string,HMMCacheEntry> entries_;
public:
 HMMCache(size_t states,size_t markers,size_t posterior):
  states_(states),markers_(markers),posterior_(posterior) {
  size_t bytes=8*(states+markers+posterior)+markers+512;
  capacity_=std::max(size_t(1),std::min(size_t(64),size_t(128*1024*1024)/std::max(size_t(1),bytes)));
 }
 HMMCacheEntry& get(const std::string& key,bool& hit){
  auto it=entries_.find(key);hit=it!=entries_.end();
  if(!hit){
   if(entries_.size()>=capacity_){
    auto oldest=entries_.begin();
    for(auto p=entries_.begin();p!=entries_.end();++p)if(p->second.used<oldest->second.used)oldest=p;
    // Every FW/scale cell is overwritten on a cache miss. Recycling the LRU
    // storage avoids allocating and zero-filling tens of MB each Gibbs update.
    auto node=entries_.extract(oldest);node.key()=key;
    node.mapped().posterior_ready=false;
    it=entries_.insert(std::move(node)).position;
   }else{
    it=entries_.try_emplace(key).first;
    it->second.fw.resize(states_);it->second.scale.resize(markers_);it->second.posterior.resize(posterior_);
   }
  }
  it->second.used=++clock_;return it->second;
 }
};
