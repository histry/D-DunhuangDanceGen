"""K-best and rank-unrank paths in precisely the original E1-W discrete DAG."""
import random
import numpy as np
from .common import identity

def lattice(interval,target,phases,beats):
 a,end=map(int,interval);A,finish=map(int,target);b=end-1;B=finish-1
 if not a<b or not A<B:raise ValueError('Two real frame centers required')
 selected=[];priority={'upper_speed_peak':0,'upper_direction_turn':1,'upper_motion_onset':2,'upper_pose_hold':3}
 for p in sorted(phases,key=lambda p:(priority.get(p['type'],9),p['frame'])):
  x=int(p['frame'])
  if a+12<x<b-12 and all(abs(x-y)>18 for y in selected):selected.append(x)
  if len(selected)==4:break
 source=[a]+sorted(selected)+[b];options=[[float(A)]];costs=[[0.]]
 for k,x in enumerate(source[1:],1):
  linear=A+(x-a)*(B-A)/(b-a);ys=[float(B)] if k==len(source)-1 else sorted(set([float(linear)]+[float(t) for t in beats if A<t<B and abs(t-linear)<=15]))
  options.append(ys);costs.append([[0. if k==len(source)-1 else min([abs(y-t) for t in beats if A<t<B] or [0.])/30,.03*((y-linear)/30)**2] for y in ys])
 # Original code adds the small linear penalty at the final endpoint, which is exactly zero.
 return dict(source=source,options=options,costs=costs)

def legal(g,k,i,j):
 rate=(g['options'][k][j]-g['options'][k-1][i])/(g['source'][k]-g['source'][k-1]);return .75-1e-10<=rate<=1.33+1e-10

def suffix(g):
 counts=[[0]*len(v) for v in g['options']];counts[-1]=[1]*len(counts[-1])
 for k in range(len(counts)-2,-1,-1):
  for i in range(len(counts[k])):counts[k][i]=sum(counts[k+1][j] for j in range(len(counts[k+1])) if legal(g,k+1,i,j))
 return counts

def best(g,K=4,fixed=None):
 fixed=fixed or {};states={0:[(0.,[])]}
 for k in range(1,len(g['source'])):
  current={}
  for j,y in enumerate(g['options'][k]):
   if k in fixed and y!=fixed[k]:continue
   paths=[(cost+g['costs'][k][j][0]+g['costs'][k][j][1],path+[[g['source'][k],y]]) for i,rows in states.items() if legal(g,k,i,j) for cost,path in rows]
   if paths:current[j]=sorted(paths,key=lambda x:(x[0],x[1]))[:K]
  states=current
 return [dict(time_warp=[[g['source'][0],g['options'][0][0]]]+p,anchor_source_frames=g['source'][1:-1],cost=float(c),rate_limits=[.75,1.33],semantic_truth='UNKNOWN',clock_contract='ACTUAL_FRAME_CENTERS_WITH_HALF_OPEN_SUPPORT') for c,p in states.get(0,[])[:K]]

def unrank(g,rank,counts=None):
 counts=suffix(g) if counts is None else counts
 if not 0<=rank<counts[0][0]:raise ValueError('Illegal path rank')
 ids=[0]
 for k in range(1,len(counts)):
  for j in range(len(counts[k])):
   n=counts[k][j] if legal(g,k,ids[-1],j) else 0
   if rank<n:ids.append(j);break
   rank-=n
 return [[x,g['options'][k][j]] for k,(x,j) in enumerate(zip(g['source'],ids))]

def rank_of(g,warp,counts):
 rank=0;previous=0
 for k,(_,y) in enumerate(warp[1:],1):
  target=g['options'][k].index(y)
  rank+=sum(counts[k][j] for j in range(target) if legal(g,k,previous,j));previous=target
 return rank

def variants(g,K,seed):
 top=best(g,K);counts=suffix(g);n=counts[0][0]
 if not top:return dict(top=[],random=[],linear=None,interior=None,legal_count=n)
 base=top[0];base_rank=rank_of(g,base['time_warp'],counts);ranks=[base_rank];rng=random.Random(seed)
 while len(ranks)<min(K,n):
  r=rng.randrange(n)
  if r not in ranks:ranks.append(r)
 def pack(w):
  cost=0.
  for k,p in enumerate(w[1:],1):
   attraction,penalty=g['costs'][k][g['options'][k].index(p[1])];cost=cost+attraction+penalty
  return dict(base,time_warp=w,cost=float(cost))
 random_paths=[pack(unrank(g,r,counts)) for r in ranks]
 fixed={1:base['time_warp'][1][1],len(g['source'])-2:base['time_warp'][-2][1]}
 interiors=[v for v in best(g,2,fixed) if v['time_warp']!=base['time_warp']]
 a,b=g['source'][0],g['source'][-1];A,B=base['time_warp'][0][1],base['time_warp'][-1][1];linear=[[x,A+(x-a)*(B-A)/(b-a)] for x in g['source']]
 return dict(top=top,random=random_paths,linear=pack(linear),interior=interiors[0] if interiors else None,legal_count=n)

def validate(warp,interval,target):
 w=np.asarray(warp,float);a,b=interval;A,B=target
 if w.ndim!=2 or w.shape[1]!=2 or not np.isfinite(w).all() or not np.array_equal(w[[0,-1]],[[a,A],[b-1,B-1]]):raise ValueError('Actual first/last frame centers must be preserved')
 d=np.diff(w,axis=0)
 if not (d>0).all() or (d[:,1]<.75*d[:,0]-1e-8).any() or (d[:,1]>1.33*d[:,0]+1e-8).any():raise ValueError('Frozen rate interval violated')
 return w
