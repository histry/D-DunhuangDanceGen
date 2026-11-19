from scipy.spatial.transform import Rotation
from .common import *
from dunhuang.research.warp_coupling_r53.population import Population
from dunhuang.research.tempo_attribution_r57.common import stats,amplification
class Assets:
 def __init__(self):
  self.pop=Population();self.old=read(PARENT/'controls/result.json');self.z=np.load(PARENT/'controls/arrays.npz');units={u['unit_uid']:u for u in read('outputs/event_coverage_r47_01/index/result.json')['units']};self.bank=[units[u] for u in self.pop.old['source_bank']];assert len(self.bank)==16;self.native={}
  for u in self.bank:
   a,b=u['source_interval'];R=self.pop.lib.rotations[u['source_uid']][a:b];P=relative_fk(R,self.pop.net.joints,self.pop.net.parents);self.native[u['unit_uid']]=stats(P)
 def record(self,song,arm):return next(r for r in self.old['records'] if r['song']==song and r['arm']==arm)
 def node(self,u,target,track,uniform=False):
  a,b=u['source_interval'];A,B=target
  if not .75-1e-10<=(B-A-1)/(b-a-1)<=1.33+1e-10:return None
  phases=[dict(p,type=p['body_region']+'_'+p['type']) for p in u['phases']];w=dict(time_warp=[[a,float(A)],[b-1,float(B-1)]],rate_limits=[.75,1.33]) if uniform else multi_warp([a,b],target,phases,track['structure']['beats'])
  if w is None:return None
  table=np.asarray(w['time_warp']);q=np.interp(np.arange(A,B),table[:,1],table[:,0]);R=rotation_sample(self.pop.lib.rotations[u['source_uid']],q);P=relative_fk(R,self.pop.net.joints,self.pop.net.parents);m=stats(P,table[1:-1,1]-A);native=self.native[u['unit_uid']];ratios=amplification(m,native);activity=float(np.linalg.norm(np.diff(P,axis=0,prepend=P[:1])*30,axis=-1).mean());energy=np.asarray(track['structure']['energy_percentile']);energy=np.pad(energy,(0,max(0,B-len(energy))),mode='edge');h=-float(abs(np.clip(activity/2,0,1)-np.mean(energy[A:B])));jets=[]
  for part in (R[:3],R[-3:]):
   anchor=part[0] if not jets else part[-1];rv=Rotation.from_matrix((anchor[None].swapaxes(-1,-2)@part).reshape(-1,3,3)).as_rotvec().reshape(3,22,3);v=(-3*rv[0]+4*rv[1]-rv[2])/2 if not jets else (3*rv[2]-4*rv[1]+rv[0])/2;jets.append(np.stack((v*30,(rv[2]-2*rv[1]+rv[0])*900)))
  data=dict(event_uid=u['unit_uid'],source_uid=u['source_uid'],source_interval=u['source_interval'],target=list(map(int,target)),warp=w,heuristic=h,activity=activity,metrics=m,native=native,source_normalized=ratios,execution_cost=execution_cost(ratios),boundary_state=np.asarray(jets).tolist(),endpoint_pose_sha256=array_hash(R[[0,-1]]),R_sha256=array_hash(R),native_sha256=self.pop.lib.rows[u['source_uid']][0]['reference']['sha256'],source_fidelity='EXACT_EVALUATION_AT_ORIGINAL_SOURCE_POSITIONS_NOT_NEURAL_RECONSTRUCTION',source_positions_sha256=array_hash(q),world_root_contact='UNKNOWN');data['id']=identity([data['event_uid'],target,w['time_warp']])[:24];return data,R

def uniform_profiles(assets):
 rows=[];track=assets.pop.tracks[0]
 for u in assets.bank:
  for length in range(CFG['min_core'],CFG['max_core']+1):
   result=assets.node(u,[0,length],track,True)
   if result is None:rows.append(dict(event_uid=u['unit_uid'],source_uid=u['source_uid'],duration=length,legal=False,reason='ORIGINAL_RATE_DOMAIN'));continue
   n,R=result;rows.append(dict(event_uid=u['unit_uid'],source_uid=u['source_uid'],duration=length,legal=True,metrics=n['metrics'],source_normalized=n['source_normalized'],activity=n['activity'],execution_cost=n['execution_cost'],boundary_state=n['boundary_state'],native_sha256=n['native_sha256']))
 return rows
