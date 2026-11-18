"""Source-verified CPU population; no encoder, diffusion or audio NN is invoked."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from .common import *
from .warp_lattice import lattice,variants,validate
from dunhuang.research.complete_r36.core import Library,multi_warp

ALGORITHMS=['src/dunhuang/research/complete_r36/core.py','src/dunhuang/research/transition_planning_r50/planning.py','src/dunhuang/research/bridge_r38/geometry.py','src/dunhuang/research/bridge_audit_r46/admission.py','src/dunhuang/research/manifold_bridge_r47/geometry.py','src/dunhuang/research/bridge_dynamics_r48/study.py','src/dunhuang/research/validation_r51/common.py']+[str(Path(__file__).with_name(n)) for n in ('common.py','population.py','warp_lattice.py','bridge_graph.py','coupled_search.py','study.py')]

def sealed(root,name):
 m=read(root/'artifacts_manifest.json')['files'];e=m[name] if isinstance(m,dict) else next(x for x in m if x['path']==name);expected=e if isinstance(e,str) else e['sha256'];p=root/name;assert sha(p)==expected;return p

class Population:
 def __init__(self):
  config=read('src/dunhuang/research/complete_r36/config.json');assert sha(config['catalog'])==config['catalog_sha256'];self.lib=Library(config['catalog']);self.old=read(R51/'controls/plans/result.json');self.r52=read(R52/'plans/result.json');assert len(self.old['records'])==CFG['registered_cases']
  assert len(self.old['source_bank'])==CFG['budget_units'];assert len(read('outputs/event_coverage_r47_01/index/result.json')['units'])==CFG['catalog_units']
  with np.load(R52/'plans/arrays.npz') as z:self.net=SimpleNamespace(joints=z['J'],parents=z['parents'])
  self.reference=read('outputs/bridge_dynamics_r48_01/main/result.json')['reference'];self.tracks=[]
  for song in range(3):
   root=Path('outputs/complete_r36_01');audio=sealed(root,f'viewer/audio/dunhuangwu{song+2}.wav');music=sealed(root,f'planning/song{song}_music.json');self.tracks.append(dict(path=str(audio),audio_sha256=sha(audio),structure=read(music),structure_sha256=sha(music)))
  self.binding=dict(protocol_sha256=sha(Path(__file__).with_name('protocol.json')),r51_population_sha256=sha(R51/'controls/plans/result.json'),r52_plans_sha256=sha(R52/'plans/result.json'),r52_arrays_sha256=sha(R52/'plans/arrays.npz'),catalog_sha256=config['catalog_sha256'],native_sources={k:v[0]['reference'] for k,v in self.lib.rows.items()},music=[{k:v for k,v in t.items() if k!='structure'} for t in self.tracks],algorithms={p:sha(p) for p in ALGORITHMS},J_sha256=array_hash(self.net.joints),parents_sha256=array_hash(self.net.parents),reference=self.reference)
  self.R={};self.nodes={}
 def node(self,c,target,warp):
  unit=c['unit'];w=validate(warp['time_warp'],unit['source_interval'],target);q=np.interp(np.arange(*target),w[:,1],w[:,0]);R=rotation_sample(self.lib.rotations[unit['source_uid']],q)
  binding=dict(source_uid=unit['source_uid'],event_uid=unit['unit_uid'],native_sha256=self.lib.rows[unit['source_uid']][0]['reference']['sha256'],source_interval=unit['source_interval'],target_interval=target,music_sha256=self._music_sha,structure_sha256=self._structure_sha,time_warp=warp['time_warp'],source_observation='COMPLETE_NATIVE_BODY22_SUPPORT; WORLD_ROOT_CONTACT_UNKNOWN')
  key=identity(binding);P=relative_fk(R,self.net.joints,self.net.parents);activity=np.linalg.norm(np.diff(P,axis=0,prepend=P[:1])*30,axis=-1).mean(1)
  # Same expression as the frozen R51/R52 Heuristic, with no new weight or warp penalty.
  target_activity=self._activity;score=-float(abs(np.clip(activity.mean()/2,0,1)-target_activity))
  values=[]
  for part in (R[:3],R[-3:]):
   center=part[0] if len(values)==0 else part[-1];r=Rotation.from_matrix((center[None].swapaxes(-1,-2)@part).reshape(-1,3,3)).as_rotvec().reshape(3,22,3)
   velocity=(-3*r[0]+4*r[1]-r[2])/2 if len(values)==0 else (3*r[2]-4*r[1]+r[0])/2;accel=r[2]-2*r[1]+r[0];values.append(np.stack((velocity*30,accel*900)))
  peaks,_=find_peaks(activity,distance=8,prominence=max(float(np.ptp(activity))*.1,1e-8));beats=np.asarray(self._beats);beats=beats[(beats>=target[0])&(beats<target[1])]-target[0];dist=np.min(abs(peaks[:,None]-beats[None]),axis=1) if len(peaks) and len(beats) else np.array([])
  jets=np.stack(values);node=dict(id=key,**binding,warp=warp,heuristic=score,activity=float(activity.mean()),warp_penalty=float(np.mean(abs(np.log(np.diff(w[:,1])/np.diff(w[:,0]))))),beat_proxy=float(np.exp(-dist**2/18).mean()) if len(dist) else None,rotation_sha256=array_hash(R),trajectory_signature_1e10=array_hash(np.rint(R/1e-10).astype(np.int64)),endpoint_jets=jets.tolist(),endpoint_rates=[float(x) for x in (np.diff(w[:,1])/np.diff(w[:,0]))[[0,-1]]])
  self.R[key]=R;self.nodes[key]=node;return key
 def banks(self,row,controls=False):
  result=[];cores=[s['interval'] for s in row['regions'] if s['kind']=='core'];track=self.tracks[row['song']];self._music_sha=track['audio_sha256'];self._structure_sha=track['structure_sha256'];self._beats=track['structure']['beats']
  for slot,(pool,target) in enumerate(zip(row['pools'],cores)):
   assert len(pool)==CFG['legal_units_per_slot'];self._activity=float(np.mean(track['structure']['energy_percentile'][slice(*target)]));bank=[]
   for c in pool:
    phases=[dict(p,type=p['body_region']+'_'+p['type']) for p in c['unit']['phases']];g=lattice(c['unit']['source_interval'],target,phases,track['structure']['beats']);paths=variants(g,CFG['K'],int(identity([CFG['seed'],row['key'],slot,c['unit']['unit_uid']])[:16],16));assert paths['top'][0]==c['warp']==multi_warp(c['unit']['source_interval'],target,phases,track['structure']['beats'])
    sets={name:[self.node(c,target,w) for w in ws] for name,ws in [('top',paths['top']),('random',paths['random'] if controls else [])]};sets['linear']=[self.node(c,target,paths['linear'])] if controls else [];sets['interior']=[self.node(c,target,paths['interior'])] if paths['interior'] else [];base=sets['top'][0];assert self.nodes[base]['heuristic']==c['parts'][0]['heuristic']
    R0=self.R[base];delta=[float(np.max(abs(self.R[k]-R0))) for k in sets['top']];jet=[float(np.max(abs(np.asarray(self.nodes[k]['endpoint_jets'])-np.asarray(self.nodes[base]['endpoint_jets'])))) for k in sets['top']]
    for k in set(sum(sets.values(),[])):assert np.array_equal(self.R[k][[0,-1]],R0[[0,-1]])
    negative=None
    if sets['interior']:
     k=sets['interior'][0]
     if not np.array_equal(self.R[k][np.r_[:4,len(R0)-4:len(R0)]],R0[np.r_[:4,len(R0)-4:len(R0)]]):raise ValueError('Interior control changed the four-frame bridge read set')
     negative=k if np.max(abs(self.R[k]-R0))>CFG['numerical_motion_tolerance'] else None
    bank.append(dict(event_uid=c['unit']['unit_uid'],source_uid=c['unit']['source_uid'],sets=sets,legal_warp_count=paths['legal_count'],distinct_warp_count=len({self.nodes[k]['trajectory_signature_1e10'] for k in sets['top']}),max_sampled_motion_delta=max(delta),max_jet_delta=max(jet),endpoint_rate_delta=[(np.asarray(self.nodes[k]['endpoint_rates'])-np.asarray(self.nodes[base]['endpoint_rates'])).tolist() for k in sets['top']],start_end_pose_error=0.,negative_control=negative,negative_status='CONSTRUCTIBLE' if negative else 'NOT_CONSTRUCTIBLE_IN_ORIGINAL_LATTICE',anchor_shift_frames=[float(np.max(abs(np.asarray(self.nodes[k]['time_warp'])[:,1]-np.asarray(self.nodes[base]['time_warp'])[:,1]))) for k in sets['top']]))
   result.append(bank)
  return result
