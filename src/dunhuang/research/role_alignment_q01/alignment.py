"""Fixed correspondence hypothesis and order-respecting null controls."""
import numpy as np
from .common import CFG

def shuffled(music):
    ordered=sorted(music,key=lambda e:(e['time'],e['uid']))
    if not ordered:return []
    shift=max(1,len(ordered)//2)%len(ordered);labels=[e['role'] for e in ordered]
    return [dict(e,role=labels[(i+shift)%len(ordered)]) for i,e in enumerate(ordered)]

def score(query,music,typed=True):
    if not query or not music:return None
    q=sorted(query,key=lambda e:(e['time'],e['uid']));m=sorted(music,key=lambda e:(e['time'],e['uid']));dp=np.zeros(len(m)+1)
    for e in q:
        nxt=np.zeros_like(dp)
        for j,v in enumerate(m):
            compatible=not typed or CFG['role_map'][e['role']]==v['role'];reward=float(np.exp(-.5*((e['time']-v['time'])/CFG['sigma_seconds'])**2)) if compatible else 0.
            nxt[j+1]=max(nxt[j],dp[j+1],dp[j]+reward)
        dp=nxt
    return float(dp[-1]/len(q))

def options(nodes,music,interval,T,start):
    a,b=interval;out=[]
    for e in nodes:
        center=(e['frame']-a)*(T-1)/(b-a-1);pool=[dict(v,local_frame=v['time']*30-start) for v in music if abs(v['time']*30-start-center)<=15 and start/30<v['time']<(start+T-1)/30]
        nearest=lambda values:sorted(values,key=lambda v:(abs(v['local_frame']-center),v['time'],v['uid']))[:2]
        union={v['uid']:v for v in nearest(pool)+nearest([v for v in pool if v['role']==CFG['role_map'][e['role']]])};out.append(sorted(union.values(),key=lambda e:(e['time'],e['uid'])))
    return out
