"""Exact-length layout, legal E1-W nodes and screened G0/G1 whole-song paths."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,subprocess
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,sha,multi_warp,rotation_sample,relative_fk
from dunhuang.research.warp_coupling_r53.common import save_arrays,array_hash
from dunhuang.research.warp_coupling_r53.population import Population
from dunhuang.research.transition_planning_r50.planning import bridge
from dunhuang.research.transition_planning_r50.common import layout
from dunhuang.research.validation_r51.common import regions
ROOT=Path('outputs/whole_song_geometry_01')
CFG=read(Path(__file__).with_name('protocol.json'))

def fresh(p):
 p=Path(p)
 if p.exists():raise ValueError('Fresh whole-song output required')
 p.mkdir(parents=True);return p

def exact_layout(frames,preferred=240,gap=48):
 if frames<2:raise ValueError('Nonempty real audio support required')
 options=[n for n in range(1,1+frames//2) if frames-(n-1)*gap>=2*n]
 count=min(options,key=lambda n:(abs((frames-(n-1)*gap)/n-preferred),n));total=frames-(count-1)*gap;small,extra=divmod(total,count);durations=[small+(i<extra) for i in range(count)]
 return layout(0,durations,[gap]*(count-1))

def coverage(spans,frames):
 count=np.zeros(frames,dtype=np.int32)
 for r in spans:
  a,b=r['interval']
  if not 0<=a<b<=frames:raise ValueError('Interval outside whole-song clock')
  count[a:b]+=1
 if not np.all(count==1):raise ValueError('Missing or overlapping music frames')
 return count

def edge_job(args):
 slot,i,j,left,right,J,parents,reference=args;net=SimpleNamespace(joints=J,parents=parents);curve,evidence=bridge(left,right,CFG['transition_frames'],net,reference)
 return slot,i,j,curve,evidence

def path_search(pools,edges,reference):
 states=[dict(score=0.,ids=[],units={},sources={})];layers=[]
 for slot,pool in enumerate(pools):
  choices=[]
  for state in states:
   for i,node in enumerate(pool):
    uid=node['unit']['unit_uid'];source=node['unit']['source_uid'];penalty=0.
    if slot:
     edge=edges.get((slot-1,state['ids'][-1],i))
     if edge is None or edge[0] is None:continue
     info=edge[1];penalty+=CFG['edge_weight']*info['methods'][info['selected']]['gap_D3']/max(reference['D3'],1e-9)
    penalty+=CFG['unit_reuse_penalty']*state['units'].get(uid,0)+CFG['source_reuse_penalty']*state['sources'].get(source,0)
    units=dict(state['units']);units[uid]=units.get(uid,0)+1;sources=dict(state['sources']);sources[source]=sources.get(source,0)+1
    choices.append(dict(score=state['score']+node['heuristic']-penalty,ids=state['ids']+[i],units=units,sources=sources))
  choices.sort(key=lambda x:(-x['score'],x['ids']));states=choices[:CFG['beam']];layers.append(dict(slot=slot,extensions=len(choices),retained=len(states)))
  if not states:return None,layers
 return states[0],layers

def run(out,render=True):
 out=fresh(out);pop=Population();track=pop.tracks[CFG['song']];structure=track['structure'];info=sf.info(track['path']);frames=int(np.ceil(info.frames*30/info.samplerate));assert abs(frames-structure['frames'])<=1
 # Every frame follows the audio; frozen music descriptors may lack a final partial frame.
 energy=np.array(structure['energy_percentile']);assert abs(len(energy)-frames)<=1
 if len(energy)<frames:energy=np.r_[energy,energy[-1]]
 spans=exact_layout(frames,CFG['preferred_core_frames'],CFG['transition_frames']);covered=coverage(spans,frames);units=read('outputs/event_coverage_r47_01/index/result.json')['units'];by={u['unit_uid']:u for u in units};bank=[by[uid] for uid in pop.old['source_bank']];assert len(bank)==16
 pools=[];rejections=[];arrays=dict(J=pop.net.joints,parents=pop.net.parents,coverage=covered);cores=[r['interval'] for r in spans if r['kind']=='core'];beats=structure['beats']
 for slot,(a,b) in enumerate(cores):
  pool=[]
  for unit in bank:
   phases=[dict(p,type=p['body_region']+'_'+p['type']) for p in unit['phases']];warp=multi_warp(unit['source_interval'],[a,b],phases,beats)
   if warp is None:rejections.append(dict(slot=slot,event_uid=unit['unit_uid'],reason='ORIGINAL_E1W_RATE_OR_ANCHOR_INFEASIBLE'));continue
   w=np.array(warp['time_warp']);q=np.interp(np.arange(a,b),w[:,1],w[:,0]);R=rotation_sample(pop.lib.rotations[unit['source_uid']],q);P=relative_fk(R,pop.net.joints,pop.net.parents);activity=np.linalg.norm(np.diff(P,axis=0,prepend=P[:1])*30,axis=-1).mean(1);h=-float(abs(np.clip(activity.mean()/2,0,1)-np.mean(energy[a:b])))
   pool.append(dict(unit=unit,warp=warp,R=R,source_positions=q,heuristic=h,activity=float(activity.mean())))
  pools.append(sorted(pool,key=lambda c:(-c['heuristic'],c['unit']['unit_uid'])));print('SONG_POOL',slot,len(pool),flush=True)
 jobs=[]
 for k in range(len(pools)-1):
  for i,l in enumerate(pools[k]):
   for j,r in enumerate(pools[k+1]):
    if not CFG['same_source_adjacent'] and l['unit']['source_uid']==r['unit']['source_uid']:continue
    jobs.append((k,i,j,l['R'],r['R'],pop.net.joints,pop.net.parents,pop.reference))
 edges={};edge_records=[]
 with ProcessPoolExecutor(max_workers=CFG['workers']) as executor:
  for k,i,j,curve,evidence in executor.map(edge_job,jobs,chunksize=8):
   edges[k,i,j]=(curve,evidence);edge_records.append(dict(slot=k,left=i,right=j,**evidence))
   if len(edges)%100==0:print('SONG_EDGES',len(edges),len(jobs),flush=True)
 choice,layers=path_search(pools,edges,pop.reference);binding=dict(parent=pop.binding,protocol_sha256=sha(Path(__file__).with_name('protocol.json')),code={p.name:sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.json')},commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_bank=pop.old['source_bank'])
 result=dict(status='NO_PATH_IN_REGISTERED_FINITE_SEARCH',audio=track['path'],audio_sha256=track['audio_sha256'],audio_samples=info.frames,sample_rate=info.samplerate,frames=frames,fps=30,duration_seconds=info.frames/info.samplerate,regions=spans,candidate_counts=list(map(len,pools)),candidate_rejections=rejections,edges=edge_records,search_layers=layers,binding=binding,P0_time_coverage=bool(np.all(covered==1)),P1_all_connections=False,P2_complete_readout=False,P3_native24='NOT_EVALUABLE',root_contact='UNKNOWN',model_calls=0,optimizer_calls=0,semantic_truth='UNKNOWN',physical_qualified=False,can_route=False)
 result['pools']=[[{k:v for k,v in node.items() if k not in ('R','source_positions')} for node in pool] for pool in pools]
 if choice is None:
  result['failure_layer']=len(layers)-1;write(out/'result.json',result);save_arrays(out/'arrays.npz',arrays);return result
 chunks=[];selected=[];connections=[];source_map=np.full(frames,np.nan);source_event=np.full(frames,-1,dtype=np.int32)
 for slot,i in enumerate(choice['ids']):
  node=pools[slot][i];chunks.append(node['R']);a,b=cores[slot];source_map[a:b]=node['source_positions'];source_event[a:b]=slot
  selected.append(dict(event_uid=node['unit']['unit_uid'],source_uid=node['unit']['source_uid'],source_interval=node['unit']['source_interval'],target_interval=[a,b],time_warp=node['warp'],heuristic=node['heuristic'],source_complete_evidence=node['unit']['stage_evidence_truncated'],native_sha256=pop.lib.rows[node['unit']['source_uid']][0]['reference']['sha256']))
  if slot<len(pools)-1:
   curve,evidence=edges[slot,i,choice['ids'][slot+1]];assert curve is not None;chunks.append(curve);connections.append(evidence)
 R=np.concatenate(chunks);assert len(R)==frames and np.isfinite(R).all();P=relative_fk(R,pop.net.joints,pop.net.parents).astype('<f4');core_errors=[]
 for k,e in enumerate(selected):
  a,b=e['target_interval'];Q=rotation_sample(pop.lib.rotations[e['source_uid']],source_map[a:b]);delta=Rotation.from_matrix((Q.swapaxes(-1,-2)@R[a:b]).reshape(-1,3,3)).magnitude();core_errors.append(float(delta.max()))
 activity=np.linalg.norm(np.diff(P.astype(float),axis=0,prepend=P[:1])*30,axis=-1).mean(1);peaks,_=find_peaks(activity,distance=8,prominence=max(float(np.ptp(activity))*.1,1e-8));beats=np.array(beats);dist=np.min(abs(peaks[:,None]-beats[None]),axis=1) if len(peaks) and len(beats) else np.array([])
 arrays.update(rotations=R,positions_root_relative=P,source_frame=source_map,source_event=source_event,world_root_translation=np.full((frames,3),np.nan),world_root_observed=np.zeros(frames,dtype=bool),contact_observed=np.zeros((frames,4),dtype=bool),body_activity=activity)
 result.update(status='GEOMETRY_WHOLE_SONG_RESEARCH_PREVIEW',selected=selected,connections=connections,path_score=choice['score'],unit_histogram=choice['units'],source_histogram=choice['sources'],P1_all_connections=True,P2_complete_readout=True,kinematics=regions(R,pop.net.joints,pop.net.parents,spans,origin=0),source_core_max_SO3_rad=core_errors,beat_proxy=dict(kinematic_peaks=len(peaks),nearest_beat_error_frames=dist.tolist(),mean_error_seconds=float(dist.mean()/30) if len(dist) else None,scope='KINEMATIC_PROXY_NOT_SEMANTIC_ALIGNMENT'),Root_world='UNKNOWN_NOT_ZERO_GROUND_TRUTH',native24='NOT_EVALUABLE_COMMON22_ONLY')
 save_arrays(out/'arrays.npz',arrays);write(out/'result.json',result)
 if render:
  from .view import run as view
  view(out,result,P,pop.net.parents)
 print('WHOLE_SONG',result['status'],frames,len(selected),len(connections),flush=True);return result

def verify(out,source):
 out=fresh(out)
 from dunhuang.research.coverage_r34.common import forbid_optimizers
 forbid_optimizers();run(out/'rebuild',render=False);old=Path(source);a=read(old/'result.json');b=read(out/'rebuild/result.json');assert a==b;assert sha(old/'arrays.npz')==sha(out/'rebuild/arrays.npz')
 write(out/'result.json',dict(status='FRESH_SOURCE_FULL_GRAPH_AND_TRAJECTORY_REPLAY_EXACT',result_sha256=sha(old/'result.json'),arrays_sha256=sha(old/'arrays.npz'),optimizer_calls=0,model_calls=0,physical_qualified=False,can_route=False))
