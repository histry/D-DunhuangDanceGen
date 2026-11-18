"""Same source/warp/geometry budgets; B3 adds a transparent recurrence penalty."""
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.transform import Rotation
from dunhuang.research.whole_song_geometry.study import edge_job,path_search as frozen_search
from .common import *

def source_kernels(pop,bank):
 rotations=[];fk=[]
 for u in bank:
  a,b=u['source_interval'];R=rotation_sample(pop.lib.rotations[u['source_uid']],np.linspace(a,b-1,CFG['source_samples']));rotations.append(R[:,1:]);fk.append(relative_fk(R,pop.net.joints,pop.net.parents)[:,1:])
 d=np.zeros((len(bank),len(bank)))
 for i in range(len(bank)):
  for j in range(i):
   x=Rotation.from_matrix((rotations[i].swapaxes(-1,-2)@rotations[j]).reshape(-1,3,3)).magnitude();d[i,j]=d[j,i]=float(np.mean(x*x))
 v=d[np.triu_indices(len(d),1)];scale=float(np.median(v[v>1e-12]));return np.exp(-d/max(scale,1e-12)),kernel(np.asarray(fk))

def guarded_edge_job(args):
 try:return edge_job(args)
 except ValueError as error:
  if str(error)!='Ambiguous SO3 chart':raise
  slot,i,j,*_=args
  return slot,i,j,None,dict(selected=None,methods={},frames=CFG['gap'],physical_qualified=False,rejection='G0_AMBIGUOUS_SO3_CHART_CURRENT_METHOD_REJECTED',G1='NOT_ATTEMPTED_AFTER_FROZEN_G0_ERROR',global_infeasibility_claim=False)


def build(pop,bank,spans,track):
 energy=np.array(track['structure']['energy_percentile']);frames=spans[-1]['interval'][1]
 if len(energy)<frames:assert len(energy)==frames-1;energy=np.r_[energy,energy[-1]]
 pools=[];rejections=[]
 for slot,r in enumerate(r for r in spans if r['kind']=='core'):
  a,b=r['interval'];pool=[]
  for index,u in enumerate(bank):
   phases=[dict(p,type=p['body_region']+'_'+p['type']) for p in u['phases']];warp=multi_warp(u['source_interval'],[a,b],phases,track['structure']['beats'])
   if warp is None:rejections.append(dict(slot=slot,event_uid=u['unit_uid'],reason='ORIGINAL_E1W_INFEASIBLE'));continue
   w=np.asarray(warp['time_warp']);q=np.interp(np.arange(a,b),w[:,1],w[:,0]);R=rotation_sample(pop.lib.rotations[u['source_uid']],q);P=relative_fk(R,pop.net.joints,pop.net.parents);activity=np.linalg.norm(np.diff(P,axis=0,prepend=P[:1])*30,axis=-1).mean(1);h=-float(abs(np.clip(activity.mean()/2,0,1)-np.mean(energy[a:b])))
   pool.append(dict(unit=u,bank=index,warp=warp,R=R,source_positions=q,heuristic=h,activity=float(activity.mean())))
  pools.append(sorted(pool,key=lambda c:(-c['heuristic'],c['unit']['unit_uid'])))
 jobs=[]
 for k in range(len(pools)-1):
  for i,left in enumerate(pools[k]):
   for j,right in enumerate(pools[k+1]):
    if left['unit']['source_uid']==right['unit']['source_uid']:continue
    jobs.append((k,i,j,left['R'],right['R'],pop.net.joints,pop.net.parents,pop.reference))
 assert len(jobs)<=CFG['maximum_geometry_trials_per_layout'];edges={};ledger=[]
 with ProcessPoolExecutor(max_workers=CFG['workers']) as executor:
  for k,i,j,curve,evidence in executor.map(guarded_edge_job,jobs,chunksize=8):
   edges[k,i,j]=(curve,evidence);ledger.append(dict(slot=k,left=i,right=j,**evidence))
 return pools,edges,dict(pools=[[{k:v for k,v in n.items() if k not in ('R','source_positions')} for n in pool] for pool in pools],edges=ledger,rejections=rejections,geometry_trials=len(jobs),eligible=sum(q[0] is not None for q in edges.values()),edge_ledger_sha256=identity(ledger),candidate_counts=list(map(len,pools)))

def choose(pools,edges,reference,music_similarity=None,motion_similarity=None):
 if music_similarity is None:return frozen_search(pools,edges,reference)
 states=[dict(score=0.,ids=[],units={},sources={})];layers=[]
 for slot,pool in enumerate(pools):
  extensions=[]
  for state in states:
   for i,node in enumerate(pool):
    u=node['unit']['unit_uid'];source=node['unit']['source_uid'];penalty=0.
    if slot:
     e=edges.get((slot-1,state['ids'][-1],i))
     if e is None or e[0] is None:continue
     info=e[1];penalty+=CFG['edge_weight']*info['methods'][info['selected']]['gap_D3']/max(reference['D3'],1e-9)
     mismatch=[(music_similarity[slot,k]-motion_similarity[node['bank'],pools[k][j]['bank']])**2 for k,j in enumerate(state['ids'])];penalty+=CFG['structure_weight']*float(np.mean(mismatch))
    penalty+=CFG['unit_reuse_penalty']*state['units'].get(u,0)+CFG['source_reuse_penalty']*state['sources'].get(source,0);units=dict(state['units']);units[u]=units.get(u,0)+1;sources=dict(state['sources']);sources[source]=sources.get(source,0)+1;extensions.append(dict(score=state['score']+node['heuristic']-penalty,ids=state['ids']+[i],units=units,sources=sources))
  extensions.sort(key=lambda x:(-x['score'],x['ids']));states=extensions[:CFG['beam']];layers.append(dict(slot=slot,extensions=len(extensions),retained=len(states)))
  if not states:return None,layers
 return states[0],layers

def assemble(pop,pools,edges,choice,spans):
 if choice is None:return None,dict(status='NO_PATH_IN_REGISTERED_FINITE_SEARCH')
 chunks=[];events=[];connections=[];frames=spans[-1]['interval'][1];source_frame=np.full(frames,np.nan);source_event=np.full(frames,-1,dtype=np.int32);cores=[r['interval'] for r in spans if r['kind']=='core']
 for k,i in enumerate(choice['ids']):
  n=pools[k][i];chunks.append(n['R']);a,b=cores[k];source_frame[a:b]=n['source_positions'];source_event[a:b]=k;events.append(dict(event_uid=n['unit']['unit_uid'],source_uid=n['unit']['source_uid'],source_interval=n['unit']['source_interval'],target_interval=[a,b],time_warp=n['warp'],bank=n['bank'],phase_evidence_truncated_count=n['unit']['stage_evidence_truncated']))
  if k<len(cores)-1:
   curve,e=edges[k,i,choice['ids'][k+1]];assert curve is not None;chunks.append(curve);connections.append(e)
 R=np.concatenate(chunks);P=relative_fk(R,pop.net.joints,pop.net.parents).astype('<f4');assert len(R)==frames and np.isfinite(R).all();arrays=dict(R=R,P=P,source_frame=source_frame,source_event=source_event,world_root_translation=np.full((frames,3),np.nan),world_root_observed=np.zeros(frames,bool))
 return arrays,dict(status='NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW',events=events,connections=connections,score=choice['score'],unit_histogram=choice['units'],source_histogram=choice['sources'],P0_time_coverage=True,P1_selected_connections=True,world_root_contact='UNKNOWN',native24='NOT_EVALUABLE')
