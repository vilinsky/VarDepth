# distutils: language = c++
"""Bulk native CIGAR queries on pysam-owned immutable BAM records."""
from collections import defaultdict
from libc.stdint cimport int64_t, int32_t, uint32_t, uint8_t
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from pysam.libcalignedsegment cimport AlignedSegment
from pysam.libchtslib cimport bam_get_cigar,bam_get_seq,bam_get_qual,bam_seqi

cdef extern from "native_reads/cigar_targets.hpp" namespace "vardepth":
    cdef cppclass TargetHit:
        int64_t position
        int32_t query
    cppbool target_hits(const uint32_t*,uint32_t,int64_t,int32_t,const vector[int64_t]&,vector[TargetHit]&) except +
    cdef cppclass QueryBounds:
        int64_t left,right,query_left,query_right
        cppbool found
    cppbool window_query(const uint32_t*,uint32_t,int64_t,int64_t,int64_t,QueryBounds&) except +
    cppbool anchor_query(const uint32_t*,uint32_t,int64_t,int64_t,const vector[int64_t]&,int64_t,QueryBounds&) except +

cdef class Positions:
    cdef vector[int64_t] values
    def __init__(self,positions):
        for p in positions:self.values.push_back(p)

def window_bounds(read,start,end):
    from alignment_queries import _window_bounds_python
    cdef AlignedSegment segment
    cdef QueryBounds out
    if isinstance(read,AlignedSegment):
        segment=read
        if window_query(bam_get_cigar(segment._delegate),segment._delegate.core.n_cigar,
                        segment._delegate.core.pos,start,end,out):
            return None if not out.found else (out.query_left,out.query_right)
    return _window_bounds_python(read,start,end)

def outer_bounds(read,start,positions,anchor,prepared=None):
    from alignment_queries import _outer_safe_anchors_python
    cdef AlignedSegment segment
    cdef QueryBounds out
    cdef Positions index
    if isinstance(read,AlignedSegment):
        segment=read;index=prepared if isinstance(prepared,Positions) else Positions(positions)
        if anchor_query(bam_get_cigar(segment._delegate),segment._delegate.core.n_cigar,
                        segment._delegate.core.pos,start,index.values,anchor,out):
            return (None,None,None,None) if not out.found else (out.left,out.right,out.query_left,out.query_right)
    return _outer_safe_anchors_python(read,start,positions,anchor)

BASES=tuple("=ACMGRSVTWYHKDBN")

def collect(reads,snps,min_bq,alignments=None,alignment_ranks=None):
    from alignment_queries import target_queries
    cdef vector[int64_t] positions
    cdef vector[TargetHit] hits
    cdef AlignedSegment segment
    cdef uint8_t* qualities
    cdef char* sequence
    cdef Py_ssize_t z
    cdef int q,bq,mq,base_index
    cdef int64_t p
    ordered=sorted(snps)
    for position in ordered:positions.push_back(position)
    observations=defaultdict(dict)
    for library,read in reads:
        key=(library,read.query_name)
        if isinstance(read,AlignedSegment):
            segment=read
            qualities=bam_get_qual(segment._delegate)
            if segment._delegate.core.l_qseq==0 or qualities[0]==255:continue
            if target_hits(bam_get_cigar(segment._delegate),segment._delegate.core.n_cigar,
                           segment._delegate.core.pos,segment._delegate.core.l_qseq,positions,hits):
                sequence=<char*>bam_get_seq(segment._delegate);mq=segment._delegate.core.qual
                for z in range(hits.size()):
                    q=hits[z].query;p=hits[z].position;bq=qualities[q]
                    if bq<min_bq:continue
                    for i in snps[p]:
                        at_site=observations[i];old=at_site.get(key)
                        if old is not None and bq<old[1]:continue
                        base_index=bam_seqi(sequence,q)
                        obs=(BASES[base_index],bq,mq)
                        if old is None or bq>old[1]:
                            at_site[key]=obs
                            if alignments is not None:alignments.setdefault(i,{})[key]=(read,q)
                        elif alignments is not None and obs==old and alignment_ranks is not None:
                            previous=alignments[i][key][0]
                            if alignment_ranks[id(read)]<alignment_ranks[id(previous)]:alignments[i][key]=(read,q)
                continue
        # Retain the original reference path for uncommon CIGARs and test doubles.
        qv=read.query_qualities
        if qv is None:continue
        for q,p in target_queries(read,ordered):
            indices=snps.get(p)
            if not indices or qv[q]<min_bq:continue
            for i in indices:
                old=observations[i].get(key);obs=(read.query_sequence[q].upper(),int(qv[q]),read.mapping_quality)
                if old is None or obs[1]>old[1]:
                    observations[i][key]=obs
                    if alignments is not None:alignments.setdefault(i,{})[key]=(read,q)
                elif alignments is not None and obs==old:
                    previous=alignments[i][key][0]
                    if alignment_ranks is not None and alignment_ranks[id(read)]<alignment_ranks[id(previous)]:alignments[i][key]=(read,q)
    return observations
