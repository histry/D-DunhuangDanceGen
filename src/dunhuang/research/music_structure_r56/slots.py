"""Exact 13-Core duration DP; a musical junction denotes a transition CENTER."""
from dunhuang.research.transition_planning_r50.common import layout
from dunhuang.research.whole_song_geometry.study import coverage,exact_layout
from .common import *

def feasible_lengths(bank):
 values=[]
 for n in range(CFG['min_core'],CFG['max_core']+1):
  if any(.75-1e-10<=(n-1)/(u['source_interval'][1]-u['source_interval'][0]-1)<=1.33+1e-10 for u in bank):values.append(n)
 return values

def allocate(frames,reward,lengths):
 count=CFG['events'];gap=CFG['gap'];total=frames-gap*(count-1);lengths=sorted(set(map(int,lengths)))
 if not lengths or not count*min(lengths)<=total<=count*max(lengths):raise ValueError('Fixed event count has no legal length budget')
 dp=np.full((count+1,total+1),-np.inf);parents=np.full((count+1,total+1),-1,dtype=np.int32);dp[0,0]=0.;target=total/count
 for k in range(1,count+1):
  for duration in lengths:
   prev=np.arange(max(0,(k-1)*min(lengths)),min(total-duration,(k-1)*max(lengths))+1);now=prev+duration;allowed=(now+(count-k)*min(lengths)<=total)&(now+(count-k)*max(lengths)>=total);prev=prev[allowed];now=now[allowed]
   if not len(prev):continue
   bonus=0. if k==count else np.asarray(reward)[now+(k-1)*gap+gap//2];value=dp[k-1,prev]+bonus-CFG['length_penalty']*((duration-target)/(CFG['max_core']-CFG['min_core']))**2;better=value>dp[k,now];dp[k,now[better]]=value[better];parents[k,now[better]]=duration
 if not np.isfinite(dp[count,total]):raise ValueError('No exact structure-duration path')
 lengths=[];cursor=total
 for k in range(count,0,-1):d=int(parents[k,cursor]);assert d>=2;lengths.append(d);cursor-=d
 spans=layout(0,lengths[::-1],[gap]*(count-1));coverage(spans,frames);return spans,float(dp[count,total])

def shuffled(spans,song):
 durations=[b-a for r in spans if r['kind']=='core' for a,b in [r['interval']]];rng=np.random.default_rng(CFG['seed']+song);order=rng.permutation(len(durations))
 if np.array_equal(order,np.arange(len(order))):order=np.roll(order,1)
 values=np.asarray(durations)[order].tolist();result=layout(0,values,[CFG['gap']]*(len(values)-1));coverage(result,spans[-1]['interval'][1]);assert sorted(values)==sorted(durations)
 return result,dict(duration_permutation=order.tolist(),same_duration_multiset=True,changed=values!=durations)

def centers(spans):return [(r['interval'][0]+r['interval'][1])//2 for r in spans if r['kind']=='transition']
