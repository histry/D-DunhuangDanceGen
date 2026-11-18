"""Unsmoothed saved FK animations, legal clocks, and complete pairwise matrices."""
import base64,json,shutil
import numpy as np
import soundfile as sf
from .common import *

def encode(x):return base64.b64encode(np.asarray(x,dtype='<f4').tobytes()).decode()
def audio(pop,song,interval,path):
 with sf.SoundFile(pop.tracks[song]['path']) as f:
  sr=f.samplerate;a,b=interval;f.seek(round(a*sr/30));wave=f.read(round(b*sr/30)-round(a*sr/30))
 sf.write(path,wave,sr)
def manifest(out,cases):write(out/'protocol.json',dict(cases=cases,scope='BODY22_RELATIVE; NO_RENDER_SMOOTHING; REJECTED_GAPS_BLANK_NOT_FILLED',files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.name!='protocol.json']))

def preview(out,pop,graph,pairs):
 folder=out/'viewer';folder.mkdir();index=[]
 for i,pair in enumerate(pairs):
  key=pair['key'];a,b=pair['music_interval'];audio(pop,pair['song'],[a,b],folder/(key+'.wav'));cells=[]
  for cell in pair['cells']:
   l=pair['lefts'][cell['left']];r=pair['rights'][cell['right']];bridge=graph.curves.get(cell['bridge_id']);valid=np.ones(b-a,bool);gap=pair['gap']
   if bridge is None:bridge=np.broadcast_to(np.eye(3),(gap,22,3,3));valid[240:240+gap]=False
   R=np.concatenate((pop.R[l],bridge,pop.R[r]));P=relative_fk(R,pop.net.joints,pop.net.parents);e=graph.records[cell['bridge_id']];cells.append(dict(**cell,data=encode(P),valid=valid.tolist(),evidence=e,left_node=pop.nodes[l],right_node=pop.nodes[r]))
  data=dict(key=key,frames=b-a,origin=a,gap=pair['gap'],audio=key+'.wav',parents=pop.net.parents.tolist(),cells=cells,default_cell=pair['default_cell'],left_count=len(pair['lefts']),right_count=len(pair['rights']),negative_controls=pair['negative_controls'])
  name=f'case{i}.js';(folder/name).write_text('window.R53_CASE='+json.dumps(data,ensure_ascii=False)+';onCaseReady();',encoding='utf8');index.append(dict(label=key,file=name))
 (folder/'data.js').write_text('window.R53_INDEX='+json.dumps(index)+';',encoding='utf8');shutil.copyfile(Path(__file__).with_name('viewer.html'),folder/'index.html');manifest(folder,len(index))

def plans(out,pop):
 # A separate global-result viewer uses the already verified R52 playback controls.
 from dunhuang.research.strong_baseline_r52.view import finish,encode
 folder=out/'viewer';folder.mkdir();cases=[]
 with np.load(out/'arrays.npz') as z:
  for row in read(out/'result.json')['records']:
   key=row['key'];audio(pop,row['song'],row['music_interval'],folder/(key+'.wav'));methods=[]
   for arm,s in row['plans'].items():
    if s['status']!='NUMERICAL_SHORT_PLAN_NOT_NATIVE':continue
    P=z[key+'_'+arm];methods.append(dict(name=arm,data=encode(P),jerk=np.r_[np.zeros(3),np.sqrt(np.mean((np.diff(P.astype(float),n=3,axis=0)*27000)**2,axis=(1,2)))].tolist()))
   cases.append(dict(label=key,frames=np.diff(row['music_interval']).item(),origin=row['music_interval'][0],audio=key+'.wav',methods=methods,meta=dict(regions=row['regions'],plans=row['plans'],scope='R53 同一候选人口、原Heuristic及原G0/G1；只改变合法时间映射。无神经生成，Root/Contact未知。')))
  finish(folder,cases,z['parents'])
 text=(folder/'index.html').read_text(encoding='utf8').replace('R52','R53').replace('保留 Heuristic 强基线；生成比较同一完整来源与几何基线，修复评价时间坐标。','冻结 Heuristic 强基线；比较固定Warp和时间—连接联合选择。没有神经训练或Diffusion。');(folder/'index.html').write_text(text,encoding='utf8');manifest(folder,len(cases))
