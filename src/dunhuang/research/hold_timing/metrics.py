"""Cell-based body-relative low activity, never semantic stillness or world physics."""
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import relative_fk
from dunhuang.research.paired_event_r41.events import spans
REGIONS={'body':tuple(range(22)),'upper':(12,13,14,15,16,17,18,19,20,21)}

def signals(R,J,parents):
 R=np.asarray(R,float);P=relative_fk(R,J,parents)
 w=Rotation.from_matrix((R[:-1].swapaxes(-1,-2)@R[1:]).reshape(-1,3,3)).as_rotvec().reshape(-1,22,3)*30
 v=np.diff(P,axis=0)*30;out={}
 for name,ids in REGIONS.items():
  out[name]=dict(angular=np.sqrt(np.mean(np.sum(w[:,ids]**2,axis=-1),axis=1)),linear=np.sqrt(np.mean(np.sum(v[:,ids]**2,axis=-1),axis=1)))
 return out,P

def low_mask(signal,threshold):return (signal['angular']<=threshold['angular']+1e-12)&(signal['linear']<=threshold['linear']+1e-12)

def describe(R,P,signal,threshold,min_cells=30):
 mask=low_mask(signal,threshold);runs=[]
 for a,b in spans(mask,min_cells):
  d=Rotation.from_matrix((R[a].swapaxes(-1,-2)[None]@R[a:b+1]).reshape(-1,3,3)).as_rotvec().reshape(-1,22,3)
  runs.append(dict(cells_half_open=[a,b],duration_seconds=(b-a)/30,angular_excursion_max_rad=float(np.linalg.norm(d,axis=-1).max()),FK_excursion_rms_max_m=float(np.sqrt(np.mean(np.sum((P[a:b+1]-P[a])**2,axis=-1),axis=-1)).max()),left_censored=a==0,right_censored=b==len(mask),semantic_truth='UNKNOWN'))
 allspans=spans(mask,1);jerk=np.linalg.norm(np.diff(P,n=3,axis=0)*30**3,axis=-1).mean(-1)
 return dict(frames=len(R),time_between_centers_seconds=(len(R)-1)/30,low_cell_fraction=float(mask.mean()),longest_low_seconds=max([b-a for a,b in allspans],default=0)/30,low_runs=runs,
  FK_D3_q95=float(np.quantile(jerk,.95)),FK_D3_max=float(jerk.max()),FK_activity_path_mean_m=float(np.linalg.norm(np.diff(P,axis=0),axis=-1).sum(0).mean()),FK_motion_range_rms_m=float(np.sqrt(np.mean((P-P.mean(0))**2))),
  cell_angular=signal['angular'].tolist(),cell_linear=signal['linear'].tolist(),low_cells=mask.tolist(),world_root_contact='UNKNOWN',hand_fingers='UNOBSERVED_BODY22_WRISTS_ONLY',semantic_hold='UNKNOWN')

def occupancy(q,source_runs,source_interval):
 q=np.asarray(q,float)
 if len(q)<2 or not (np.diff(q)>0).all():raise ValueError('Strictly increasing actual source clock required')
 rows=[]
 for h in source_runs:
  a,b=h['cells_half_open'];l=max(float(a),q[0]);r=min(float(b),q[-1])
  if r<=l:continue
  t=np.interp([l,r],q,np.arange(len(q)))
  rows.append(dict(source_cells_half_open=[a,b],source_supported_frame_centers=[l,r],target_frame_centers=t.tolist(),natural_supported_seconds=(r-l)/30,target_seconds=float((t[1]-t[0])/30),
   minimum_seconds_at_rate_075=.75*(r-l)/30,maximum_seconds_at_rate_133=1.33*(r-l)/30,clipped_by_core=bool(l>a or r<b),source_role='AUTOMATIC_LOW_ACTIVITY_NOT_ARTISTIC_HOLD'))
 return rows

def excess(holds,cap,music=None,origin=0):
 if cap is None:return None
 total=0.
 for h in holds:
  allowed=cap
  if music is not None:
   l,r=h['target_frame_centers'];a=max(0,int(np.floor(origin+l)));b=min(len(music),int(np.ceil(origin+r))+1)
   if b<=a:return None
   activity=float(np.mean(music[a:b]));allowed*=1.33-(1.33-.75)*activity
  total+=max(0,h['target_seconds']-allowed)**2
 return float(total)
