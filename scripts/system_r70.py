"""R70 one-song system preview: cached full planner versus fresh official EDGE.

ChoreoGraph is unavailable, never substituted by R69's DTW adaptation.
"""
import argparse,json,hashlib,subprocess,sys,shutil
from pathlib import Path
import numpy as np
from dunhuang.contracts import MotionAdapter,matrices
from dunhuang.research.complete_r36.core import relative_fk
from dunhuang.research.warp_coupling_r53.common import save_arrays,array_hash

GEO=Path('outputs/whole_song_geometry_02/main')
ROOT=Path('outputs/system_r70_01')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())
def native_points(edge,J,parents,frames):
 if len(edge)<frames:raise ValueError('Generator did not cover the music')
 x=MotionAdapter.to_native(np.asarray(edge,dtype=np.float32)[:frames]);R=matrices(x[:,7:].reshape(-1,24,6))[:,:22]
 return relative_fk(R,J,parents)
def run(out,replay=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 assert sha(GEO/'result.json')=='bb0907101a2fa8d308de2b7ed4ac8fdcc5b2e4f3c8c6a28020bb58773440106d'
 assert sha(GEO/'arrays.npz')=='3a1d39cc1fd7cec7ced8dd5d46730ef2f418bd639ad607e8bdbca724882e5582'
 z=np.load(GEO/'arrays.npz');frames=len(z['rotations']);assert frames==3601
 q=json.loads(Path('outputs/exposed_full_song_01/generation/0_baseline/request.json').read_text())
 assert q['audio']['sha256']=='36ce6f03d7a8c57507071fcd74429562b256c03f5523c7e0657f0d1037cd598c'
 assert sha(q['audio']['path'])==q['audio']['sha256'];assert sha(q['features']['path'])==q['features']['sha256']
 q.update(mode='baseline',seed=70070,anchors=[],output=str((out/'edge').resolve()),frames=frames)
 write(out/'request.json',q)
 subprocess.run([sys.executable,'-m','dunhuang.edge_backend',str(out/'request.json')],check=True)
 edge=np.load(out/'edge/edge.npy');P=native_points(edge,z['J'],z['parents'],frames)
 current=relative_fk(z['rotations'],z['J'],z['parents'])
 save_arrays(out/'positions.npz',dict(current=current,edge=P))
 gen=json.loads((out/'edge/generation.json').read_text());assert gen['sampler']['method']=='official_long_ddim' and gen['weights_before']==gen['weights_after']
 result=dict(schema='R70_ONE_EXPOSED_SONG_SYSTEM_PREVIEW_V1',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),frames=frames,fps=30,
  music=q['audio'],seed=q['seed'],checkpoint=q['checkpoint'],features=q['features'],official_edge_commit=q['edge_commit'],sampler=gen['sampler'],
  systems=dict(current=dict(status='FROZEN_FULL_SONG_SYSTEM_REUSE',source_result_sha256=sha(GEO/'result.json'),source_arrays_sha256=sha(GEO/'arrays.npz'),source_condition='ChangE16 source units; full120s heuristic E1W G0/G1 graph; not fixed R69 event selection'),
   EDGE=dict(status='FRESH_OFFICIAL_SAMPLER_REFERENCE',training_domain='AIST++ pretrained; no ChangE reference condition',frames_generated=len(edge)),
   ChoreoGraph=dict(status='NOT_REPRODUCED_OFFICIAL_CODE_AND_WEIGHTS_NOT_LOCATED',paper='https://arxiv.org/html/2207.07386',substitution=False)),
  position_hashes=dict(current=array_hash(current),EDGE=array_hash(P)),render_contract='SAME22_JOINTS_SKELETON_METRIC_SCALE_YUP_ROOT_RELATIVE_CAMERA_NO_PER_ARM_RESCALE',
  native24='NOT_EVALUABLE',human_reviewed=False,quality_winner=None,scope='ONE_HISTORICALLY_EXPOSED_MUSIC_SYSTEM_PRECOMPARISON_NOT_THREE_METHOD_BENCHMARK',training_steps=0)
 write(out/'result.json',result)
 if replay:
  old=json.loads((Path(replay)/'result.json').read_text());exact=result['position_hashes']==old['position_hashes'];write(out/'independent.json',dict(exact=exact,source_result_sha256=sha(Path(replay)/'result.json'),fresh_edge_sampling=True));assert exact
 else:
  watch=out/'viewer';watch.mkdir();shutil.copyfile(q['audio']['path'],watch/'music.wav')
  payload=dict(parents=z['parents'].tolist(),frames=frames,arms=[dict(name='当前系统 · Heuristic＋E1-W＋Geometry',P=np.round(current,5).tolist()),dict(name='官方 EDGE · 无 ChangE 条件',P=np.round(P,5).tolist()),dict(name='ChoreoGraph · 尚未复现',P=None)])
  html=Path(__file__).with_name('system_r70.html').read_text();(watch/'index.html').write_text(html.replace('__DATA__',json.dumps(payload,separators=(',',':'))),encoding='utf8')
  write(out/'viewer_manifest.json',dict(files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in watch.iterdir()]))
 print('R70_COMPLETE',out,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--replay');a=p.parse_args();run(a.out,a.replay)
