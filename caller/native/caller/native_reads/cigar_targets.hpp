// Exact CIGAR-to-target projection. No scoring, filtering or candidate pruning.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>
namespace vardepth {
struct TargetHit { int64_t position; int32_t query; };
struct MatchRun { int64_t query,position,length; };
struct QueryBounds { int64_t left=-1,right=-1,query_left=-1,query_right=-1; bool found=false; };
inline bool match_runs(const uint32_t* cigar,uint32_t count,int64_t start,
                       std::vector<MatchRun>& runs) {
    runs.clear();int64_t q=0,p=start;bool pending=false;
    for(uint32_t c=0;c<count;++c) {
        unsigned op=cigar[c]&15u;int64_t n=cigar[c]>>4;
        if(op==6 || op==9 || op>9) return false;
        if(op==0 || op==7 || op==8) {
            if(pending && runs.back().query+runs.back().length==q &&
               runs.back().position+runs.back().length==p) runs.back().length+=n;
            else runs.push_back(MatchRun{q,p,n});
            pending=true;q+=n;p+=n;
        } else {
            pending=false;
            if(op==1 || op==4) q+=n;
            else if(op==2 || op==3) p+=n;
        }
    }
    return true;
}
inline bool window_query(const uint32_t* cigar,uint32_t count,int64_t start,
                          int64_t left,int64_t right,QueryBounds& out) {
    std::vector<MatchRun> runs;out=QueryBounds{};
    if(!match_runs(cigar,count,start,runs))return false;
    for(const auto& run:runs) {
        const int64_t lo=std::max(run.position,left),hi=std::min(run.position+run.length,right);
        if(lo>=hi)continue;
        if(!out.found)out.query_left=run.query+lo-run.position;
        out.found=true;
        out.query_right=run.query+hi-run.position;
    }
    return true;
}
inline bool anchor_query(const uint32_t* cigar,uint32_t count,int64_t read_start,
                         int64_t start,const std::vector<int64_t>& positions,
                         int64_t anchor,QueryBounds& out) {
    std::vector<MatchRun> runs;out=QueryBounds{};
    if(!match_runs(cigar,count,read_start,runs))return false;
    for(const auto& run:runs) {
        auto lo=std::lower_bound(positions.begin(),positions.end(),run.position-start);
        auto hi=std::upper_bound(positions.begin(),positions.end(),run.position+run.length-start-anchor);
        if(lo>=hi)continue;
        if(!out.found){out.left=*lo;out.query_left=run.query+start+out.left-run.position;}
        out.found=true;
        out.right=*(hi-1)+anchor;out.query_right=run.query+start+out.right-run.position;
    }
    return true;
}
inline bool target_hits(const uint32_t* cigar, uint32_t count, int64_t start,
                        int32_t query_length, const std::vector<int64_t>& positions,
                        std::vector<TargetHit>& out) {
    out.clear();
    for (uint32_t c=0;c<count;++c) {
        const unsigned op=cigar[c]&15u;
        if(op==6 || op==9 || op>9) return false; // Preserve pysam's uncommon-op path.
    }
    int64_t q=0,p=start;
    auto it=std::lower_bound(positions.begin(),positions.end(),p);
    for(uint32_t c=0;c<count;++c) {
        const unsigned op=cigar[c]&15u;
        const int64_t n=cigar[c]>>4;
        if(op==0 || op==7 || op==8) {
            it=std::lower_bound(it,positions.end(),p);
            while(it!=positions.end() && *it<p+n) {
                const int64_t query=q+*it-p;
                if(query<0 || query>=query_length) {out.clear();return false;}
                out.push_back(TargetHit{*it,static_cast<int32_t>(query)});
                ++it;
            }
            q+=n;p+=n;
        } else if(op==1 || op==4) q+=n;
        else if(op==2 || op==3) p+=n;
    }
    return true;
}
}
