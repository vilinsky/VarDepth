import math


def quantile(hist,q):
    n=sum(hist.values())
    if not n:return None
    rank=(n-1)*q;a=math.floor(rank);b=math.ceil(rank);total=0;va=vb=None
    for value,count in sorted(hist.items()):
        if total<=a<total+count:va=value
        if total<=b<total+count:vb=value;break
        total+=count
    assert va is not None and vb is not None
    return va+(vb-va)*(rank-a)
