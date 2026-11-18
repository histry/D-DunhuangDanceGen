"""Exact finite search, with the original Heuristic and 0.01 bridge penalty."""
from itertools import product
import numpy as np
from .common import *
from dunhuang.research.validation_r51.common import regions
from scipy.signal import find_peaks

def solve(pop,graph,layers,edges):
 ordered=[sorted(ids,key=lambda k:(-pop.nodes[k]['heuristic'],pop.nodes[k]['event_uid'],k)) for ids in layers];rank=[{k:i for i,k in enumerate(ids)} for ids in ordered];best=None;visited=0;legal_paths=0
 for ids in product(*ordered):
  visited+=1;links=[edges.get((i,l,r)) for i,(l,r) in enumerate(zip(ids,ids[1:]))]
  if any(k not in graph.curves for k in links):continue
  legal_paths+=1;score=sum(pop.nodes[k]['heuristic'] for k in ids)-CFG['bridge_penalty']*sum(graph.records[k]['methods'][graph.records[k]['selected']]['gap_D3']/max(pop.reference['D3'],1e-9) for k in links);candidate=(score,tuple(-rank[i][k] for i,k in enumerate(ids)),ids,links)
  if best is None or candidate[:2]>best[:2]:best=candidate
 return dict(status='REJECTED_REGISTERED_METHODS_ONLY',paths_visited=visited,admitted_paths=0) if best is None else dict(status='NUMERICAL_SHORT_PLAN_NOT_NATIVE',score=best[0],nodes=list(best[2]),bridges=list(best[3]),paths_visited=visited,admitted_paths=legal_paths)

def layers_for(pop,banks,mode,K=4):
 out=[]
 for layer in banks:
  ids=[]
  for event in layer:
   choices=event['sets']['random' if mode=='random' else 'linear' if mode=='linear' else 'top'][:K]
   if mode=='fixed':choices=choices[:1]
   if mode=='independent':choices=sorted(choices,key=lambda k:(-pop.nodes[k]['heuristic'],event['sets']['top'].index(k)))[:1]
   ids.extend(choices)
  out.append(ids)
 return out

def compose(pop,graph,solution):
 if solution['status']!='NUMERICAL_SHORT_PLAN_NOT_NATIVE':return None
 chunks=[]
 for i,key in enumerate(solution['nodes']):
  chunks.append(pop.R[key])
  if i<len(solution['bridges']):chunks.append(graph.curves[solution['bridges'][i]])
 return np.concatenate(chunks)

def evaluate(R,row,pop):
 result=regions(R,pop.net.joints,pop.net.parents,row['regions'],origin=row['music_interval'][0]);parts=result['regions'];cores=[p['metrics']['D3_Q95'] for p in parts if p['kind']=='core'];boundaries=[p['metrics']['D3_max'] for p in parts if p['kind']=='left_boundary'];P=relative_fk(R,pop.net.joints,pop.net.parents);speed=np.r_[0,np.sqrt(np.mean((np.diff(P,axis=0)*30)**2,axis=(1,2)))];peaks,_=find_peaks(speed,distance=8,prominence=max(float(np.ptp(speed))*.1,1e-8));a,b=row['music_interval'];beats=np.asarray(pop.tracks[row['song']]['structure']['beats']);beats=beats[(beats>=a)&(beats<b)]-a;dist=np.min(abs(peaks[:,None]-beats[None]),axis=1) if len(peaks) and len(beats) else np.array([])
 result.update(independent_boundary_D3_mean=float(np.mean(boundaries)),core_D3_Q95_mean=float(np.mean(cores)),beat_proxy=float(np.exp(-dist**2/18).mean()) if len(dist) else None,peak_to_beat_frames=dist.tolist(),arm_extent_rms_m=float(np.sqrt(np.mean(np.ptp(P[:,16:22],axis=0)**2))),metric_scope='UNOPTIMIZED_PROXY_READOUTS_NOT_INDEPENDENT_HUMAN_OR_SEMANTIC_TRUTH');return result
