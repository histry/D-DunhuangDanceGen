import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.transition_r37.evaluate import decode,world_fk
import sys
root=Path(sys.argv[1]);meta=json.loads((root/'evaluation/evaluation.json').read_text());a=np.load(root/'evaluation/arrays.npz');J=a['skeleton_J22'];parents=a['parents'];reports=[]
for row in meta['records']:
 key=row['key'];lo,hi=row['gap'];envelope=__import__('dunhuang.research.transition_r37.geometry',fromlist=['c2_envelope']).c2_envelope(256,row['gap']);cuts=sorted(set(row['cuts']+[lo,hi]));per={}
 for arm in ('Geometry','FrozenMatched','Trained'):
  cases=[]
  for i,raw in enumerate(a[key+'_'+arm+'_139']):
   R,Root=decode(raw);points=world_fk(R,Root,J,parents);saved=a[key+'_'+arm+'_22'][i];err=float(abs(points-saved).max())
   if err>2e-6:raise ValueError((key,arm,'FK decode',err))
   base=a[key+'_Geometry_R'][i];angle=Rotation.from_matrix((base.swapaxes(-1,-2)@R).reshape(-1,3,3)).magnitude().reshape(256,22)
   if np.max(angle-.6*envelope[:,None])>1e-5:raise ValueError('Projection budget violation')
   jerk=np.linalg.norm(np.diff(points,n=3,axis=0)*27000,axis=-1).mean(-1);rootacc=np.linalg.norm(np.diff(Root,n=2,axis=0)*900,axis=-1)
   cutmask=np.zeros(253,bool)
   for b in cuts:cutmask[max(0,b-3):min(b,253)]=True
   cases.append(dict(seed=row['seeds'][i],fk_max_error_m=err,all_cuts_including_observation_edges=cuts,world_jerk_mean=float(jerk.mean()),world_jerk_p95=float(np.percentile(jerk,95)),all_cut_world_jerk_peak=float(jerk[cutmask].max()),root_acceleration_peak_m_s2=float(rootacc.max()),geometry_residual_angle_mean_rad=float(angle.mean()),budget_verified=True))
  per[arm]=cases
 reports.append(dict(key=key,methods=per))
out=dict(records=reports,scope='INDEPENDENT_LOCAL_CPU_DECODE_AND_BUDGET_AUDIT',source_arrays_sha256=hashlib.sha256((root/'evaluation/arrays.npz').read_bytes()).hexdigest(),full_native24='NOT_EVALUABLE',contact_truth='UNKNOWN')
(root/'independent_local_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('CPU_FK_AND_PROJECTION_VERIFIED',len(reports)*3*2)
