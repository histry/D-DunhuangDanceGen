from pathlib import Path
import json,hashlib,subprocess
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk,rotation_sample,multi_warp
from dunhuang.research.warp_coupling_r53.common import save_arrays,array_hash
ROOT=Path('outputs/tempo_attribution_r57_01')
PARENT=Path('outputs/music_structure_r56_02')
CFG=read(Path(__file__).with_name('protocol.json'))
def fresh(path):
 p=Path(path)
 if p.exists():raise ValueError('Fresh R57 output required: '+str(p))
 p.mkdir(parents=True);return p
def binding():
 return dict(code={p.name:sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.json')},parent_manifest_sha256=sha(PARENT/'artifacts_manifest.json'),parent_result_sha256=sha(PARENT/'controls/result.json'),parent_arrays_sha256=sha(PARENT/'controls/arrays.npz'),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
def verify_parent():
 rows=read(PARENT/'artifacts_manifest.json')['files']
 for r in rows:assert sha(PARENT/r['path'])==r['sha256'],r['path']
 return len(rows)
def stats(P,knots=()):
 P=np.asarray(P,float)
 if P.ndim!=3 or P.shape[1:]!=(22,3) or len(P)<4 or not np.isfinite(P).all():raise ValueError('Observed finite body22 and four frames required')
 out=dict(frames=len(P))
 for k in (1,2,3):
  x=np.sqrt(np.mean((np.diff(P,n=k,axis=0)*30**k)**2,axis=(1,2)));out['D%d_Q95'%k]=float(np.quantile(x,.95));out['D%d_max'%k]=float(x.max())
  if k==3:
   q=np.arange(len(x));mask=np.zeros(len(x),bool)
   for t in knots:mask|=(q<t)&(q+k>=t)
   out['knot_D3']=dict(stencils=int(mask.sum()),all_stencils=len(x),Q95=float(np.quantile(x[mask],.95)) if mask.any() else None,nonknot_Q95=float(np.quantile(x[~mask],.95)) if (~mask).any() else None,squared_mass_fraction=float(np.sum(x[mask]**2)/np.sum(x**2)) if np.sum(x**2)>CFG['epsilon'] else None,definition='DIFFERENCE_STENCILS_CROSS_INTERIOR_TARGET_WARP_KNOTS')
 return out
def amplification(value,native):
 return {str(k):dict(ratio=value['D%d_Q95'%k]/native['D%d_Q95'%k] if native['D%d_Q95'%k]>CFG['epsilon'] else None,status='OBSERVED_RATIO_NOT_NATURALNESS' if native['D%d_Q95'%k]>CFG['epsilon'] else 'SOURCE_NEAR_STATIC_UNDEFINED') for k in (1,2,3)}
def shapley(cells):
 if len(cells)!=4:return dict(status='NOT_CONSTRUCTIBLE_ON_COMMON_SUPPORT')
 a,b,c,d=cells['00'],cells['01'],cells['10'],cells['11'];selection=.5*((c-a)+(d-b));timing=.5*((b-a)+(d-c));assert abs(selection+timing-(d-a))<1e-7
 return dict(status='SHARED_SUPPORT_FACTORIAL_DECOMPOSITION',selection=selection,timing=timing,total=d-a,interaction=d-c-b+a,scope='SLOT_CORE_Q95_DIFFERENCE_NOT_WHOLE_SONG_Q95_OR_PHYSICAL_CAUSALITY')
