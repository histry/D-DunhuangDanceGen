"""Frozen research contracts; no changes to the production retrieval service."""
from pathlib import Path
import json,hashlib
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.warp_coupling_r53.common import save_arrays,array_hash
ROOT=Path('outputs/residual_sdmr_r54_02')
CFG=read(Path(__file__).with_name('protocol.json'))
def fresh(p):
 p=Path(p)
 if p.exists():raise ValueError('Fresh R54 output required: '+str(p))
 p.mkdir(parents=True);return p
def sealed(root,name):
 rows=read(Path(root)/'artifacts_manifest.json')['files'];r=next(r for r in rows if r['path']==name);p=Path(root)/name
 assert sha(p)==r['sha256'],str(p)
 return p
def code_binding():return {p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py')) if p.name not in ('view.py','release.py','__main__.py')}
def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def heuristic(music_activity,motion_activity):return -np.abs(np.asarray(music_activity)-np.asarray(motion_activity))
def relation(a,b):
 if a['role']!=b['role'] or a['raw_motion_sha256']==b['raw_motion_sha256']:return 'UNKNOWN'
 A,B=a['audio_frame_interval'],b['audio_frame_interval'];inter=max(0,min(A[1],B[1])-max(A[0],B[0]));union=max(A[1],B[1])-min(A[0],B[0]);same=a['audio_sha256']==b['audio_sha256']
 if same and inter/max(union,1e-9)>=CFG['positive_iou']:return 'CLOCK_POSITIVE'
 if same and max(A[0],B[0])-min(A[1],B[1])>=CFG['negative_gap']:return 'CLOCK_NON_TARGET_TIME'
 if not same and a['choreography_uid']!=b['choreography_uid']:return 'DIFFERENT_AUDIO_ASSOCIATION'
 return 'UNKNOWN'
def permuted_labels(p,seed):
 p=np.asarray(p,bool);rng=np.random.default_rng(seed);q=p[rng.permutation(len(p))]
 if np.array_equal(p,q):q=np.roll(p,1)
 assert q.sum()==p.sum() and not np.array_equal(q,p)
 return q.tolist()

def matched_bin(h,body,reference_h,reference_body):
 return int(np.floor(h/CFG['heuristic_bin_width']))==int(np.floor(reference_h/CFG['heuristic_bin_width'])) and int(np.floor(body/CFG['motion_activity_bin_width']))==int(np.floor(reference_body/CFG['motion_activity_bin_width']))
