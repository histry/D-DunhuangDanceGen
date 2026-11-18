"""Frozen Heuristic strong baseline and matched score/structure comparisons."""
from itertools import product
import numpy as np
from .common import read,write,relative_fk,fresh
from dunhuang.research.complete_r36.core import rotation_sample
from dunhuang.research.transition_planning_r50.common import assets
from dunhuang.research.transition_planning_r50.planning import bridge
from dunhuang.research.validation_r51.common import regions


def choose(pools,edges,reference,method,no_repeat=False,balanced=False):
 def score(c):
  h=c['parts'][0]['heuristic'];s=c['scores'][0]
  return h if method=='Heuristic' else s if method=='SDMR' else h+.1*s if method=='Hybrid' else h+.1*c['parts'][0]['multi']
 choices=[]
 for pool in pools:
  ids=list(range(len(pool)))
  if balanced:
   seen=set();ids=[]
   for i in sorted(range(len(pool)),key=lambda k:(pool[k]['unit']['source_uid'],pool[k]['unit']['source_interval'])):
    uid=pool[i]['unit']['source_uid']
    if uid not in seen:seen.add(uid);ids.append(i)
  choices.append(sorted(ids,key=lambda i:(-score(pool[i]),pool[i]['unit']['unit_uid'])))
 best=None
 for ids in product(*choices):
  uids=[pools[k][i]['unit']['unit_uid'] for k,i in enumerate(ids)]
  if no_repeat and len(set(uids))<len(uids):continue
  es=[edges.get((k,ids[k],ids[k+1])) for k in range(len(ids)-1)]
  if any(e is None or e[0] is None for e in es):continue
  value=sum(score(pools[k][i]) for k,i in enumerate(ids))-.01*sum(e[1]['methods'][e[1]['selected']]['gap_D3']/max(reference['D3'],1e-9) for e in es)
  candidate=(value,tuple(-choices[k].index(i) for k,i in enumerate(ids)),ids)
  if best is None or candidate[:2]>best[:2]:best=candidate
 return None if best is None else dict(score=best[0],ids=list(best[2]))


def run(out,controls=False):
 out=fresh(out);net,lib,units,tracks=assets();old=read('outputs/validation_r51_01/controls/plans/result.json');reference=read('outputs/bridge_dynamics_r48_01/main/result.json')['reference'];arrays=dict(J=net.joints,parents=net.parents);records=[]
 for row in old['records']:
  if controls and not (row['events']==2 and row['gap']==48):continue
  pools=[];cores=[s['interval'] for s in row['regions'] if s['kind']=='core']
  for pool,(a,b) in zip(row['pools'],cores):
   pp=[]
   for c in pool:
    w=np.asarray(c['warp']['time_warp']);q=np.interp(np.arange(a,b),w[:,1],w[:,0]);pp.append(dict(c,R=rotation_sample(lib.rotations[c['unit']['source_uid']],q)))
   pools.append(pp)
  edges={(e['slot'],e['left'],e['right']):(True if e['selected'] else None,{k:v for k,v in e.items() if k not in ('slot','left','right')}) for e in row['edges']}
  extra={}
  if controls:
   for k in range(len(pools)-1):
    for i,l in enumerate(pools[k]):
     for j,r in enumerate(pools[k+1]):
      if l['unit']['source_uid']==r['unit']['source_uid'] and l['unit']['unit_uid']!=r['unit']['unit_uid'] and l['unit']['source_interval'][1]<=r['unit']['source_interval'][0]:extra[k,i,j]=bridge(l['R'],r['R'],row['gap'],net,reference)
  arms=[('Heuristic',False,False,False)] if not controls else [(method,repeat,same,balanced) for method in ('Heuristic','Hybrid','SDMR','RhythmCurve') for repeat in (False,True) for same in (False,True) for balanced in (False,True)]
  selected={}
  for method,repeat,same,balanced in arms:
   graph=edges|extra if same else edges;key=f'{method}_repeat{int(repeat)}_same{int(same)}_balanced{int(balanced)}';choice=choose(pools,graph,reference,method,repeat,balanced)
   if choice is None:selected[key]=dict(status='REJECTED_METHOD_SCREEN_NOT_PROVEN_IMPOSSIBLE');continue
   ids=choice['ids'];chunks=[];events=[]
   for k,i in enumerate(ids):
    c=pools[k][i];chunks.append(c['R']);events.append(dict(event_uid=c['unit']['unit_uid'],source_uid=c['unit']['source_uid'],source_interval=c['unit']['source_interval'],warp=c['warp']))
    if k<len(ids)-1:
     curve,evidence=bridge(c['R'],pools[k+1][ids[k+1]]['R'],row['gap'],net,reference);assert curve is not None and evidence==graph[k,i,ids[k+1]][1];chunks.append(curve)
   R=np.concatenate(chunks);name=row['key']+'_'+key;arrays[name+'_R']=R;arrays[name]=relative_fk(R,net.joints,net.parents).astype('<f4')
   selected[key]=dict(status='NUMERICAL_SHORT_PLAN',events=events,**choice,kinematics=regions(R,net.joints,net.parents,row['regions'],origin=row['music_interval'][0]))
   if method=='Heuristic' and not(repeat or same or balanced):assert events==row['selected']['HeuristicFullPool']['events'],'Frozen strong baseline changed'
  records.append(dict(key=row['key'],song=row['song'],events=row['events'],gap=row['gap'],regions=row['regions'],music_interval=row['music_interval'],selected=selected,same_source_forward_candidates=len(extra),geometry_rejection_scope='METHOD_SPECIFIC_NOT_GLOBAL_NECESSARY_CONDITION'))
  print('R52_PLAN',row['key'],len(selected),flush=True)
 np.savez_compressed(out/'arrays.npz',**arrays);write(out/'result.json',dict(records=records,weights=net.binding(),can_route=False,training_steps=0))
 from .view import plans
 plans(out,tracks)
