"""Read-only skeleton viewers use one audio clock and fixed display seed."""
import json,shutil,base64
from pathlib import Path
import numpy as np
import soundfile as sf
from .common import read,write,sha,ROOT


def encode(x):return base64.b64encode(np.asarray(x,dtype='<f4').tobytes()).decode()

def finish(out,cases,parents):
 (out/'data.js').write_text('window.R51='+json.dumps(dict(cases=cases,parents=parents.tolist()),ensure_ascii=False)+';',encoding='utf8');shutil.copyfile(Path(__file__).with_name('viewer.html'),out/'index.html');write(out/'protocol.json',dict(cases=len(cases),files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.name!='protocol.json']))


def plans(root,tracks):
 out=root/'viewer';out.mkdir();z=np.load(root/'arrays.npz');cases=[]
 for r in read(root/'result.json')['records']:
  if len(r['selected'])>1 and r['gap']!=48:continue
  a,b=r['music_interval'];name=r['key']+'.wav'
  with sf.SoundFile(tracks[r['song']]['path']) as f:
   sr=f.samplerate;f.seek(round(a*sr/30));wave=f.read(round(b*sr/30)-round(a*sr/30))
  sf.write(out/name,wave,sr);methods=[]
  for arm,s in r['selected'].items():
   if s['status']!='NUMERICAL_SHORT_PLAN':continue
   P=z[r['key']+'_'+arm];jerk=np.r_[np.zeros(3),np.sqrt(np.mean((np.diff(P.astype(float),n=3,axis=0)*27000)**2,axis=(1,2)))];methods.append(dict(name=arm,data=encode(P),jerk=jerk.tolist()))
  cases.append(dict(label=r['key'],frames=b-a,origin=a,audio=name,methods=methods,meta=dict(regions=r['regions'],selected=r['selected'],scope='冻结Heuristic来源重定时＋G0/G1；区域统计已转换局部帧。高活动不自动代表异常抖动。',root_contact='UNKNOWN')))
 finish(out,cases,z['parents'])


def generation(root):
 out=root/'viewer';out.mkdir();z=np.load(root/'arrays.npz');cases=[];baseline_path=ROOT/'frozen_full/arrays.npz';baseline=np.load(baseline_path) if baseline_path.exists() and root!=ROOT/'frozen_full' else None
 for r in read(root/'result.json')['records']:
  key=r['key'];plan=read(r['plan']);audio=key+'.wav';shutil.copyfile(plan['music']['audio']['path'],out/audio);methods=[dict(name='来源＋几何Bridge',data=encode(z[key+'_Geometry']))]
  if baseline is not None:methods.append(dict(name='Frozen Full Reference',data=encode(baseline[key+'_FullReference'][0])))
  methods += [dict(name=name,data=encode(z[key+'_'+name][0])) for name in r['methods']]
  for m in methods:
   P=np.frombuffer(base64.b64decode(m['data']),dtype='<f4').reshape(r['frames'],22,3).astype(float);m['jerk']=np.r_[np.zeros(3),np.sqrt(np.mean((np.diff(P,n=3,axis=0)*27000)**2,axis=(1,2)))].tolist()
  cases.append(dict(label=key+' · 完整双事件',frames=r['frames'],origin=r['music_interval'][0],audio=audio,methods=methods,meta=dict(regions=plan['regions'],source_events=plan['events'],methods=r['methods'],source_visibility=r['source_visibility'],scope='完整所选Core可读；全部528帧共同去噪，只有外端16帧硬保护；展示52051，统计两种子；Root/Contact未知。')))
 finish(out,cases,z['parents'])


def hub(root):
 fixtures=[];plans=read(root/'plans/result.json')['records'];n=read(root/'main/result.json')
 for i,row in enumerate(n['records']):
  plan_path=root/'interface'/(row['key']+'.json');assert sha(plan_path)==row['plan_sha256'];plan=read(plan_path);song=int(row['key'].replace('fixture',''));index=next(k for k,r in enumerate(plans) if r['song']==song and r['events']==2 and r['gap']==48);selected=next(iter(plans[index]['selected'].values()))['events'];assert [e['event_uid'] for e in selected]==[e['event_uid'] for e in plan['events']]
  assert plans[index]['music_interval']==plan['target_interval'];fixtures.append(dict(label='dunhuangwu%d · 同一双事件计划'%(song+2),frames=row['frames'],origin=plan['target_interval'][0],audio='main/viewer/'+row['key']+'.wav',plan_index=index,generation_index=i,segments=[dict(label=('动作'+str(k//2+1)) if span['kind']=='core' else '过渡',start=span['interval'][0]-plan['target_interval'][0],end=span['interval'][1]-plan['target_interval'][0]) for k,span in enumerate(plan['regions'])]))
 shutil.copyfile(Path(__file__).with_name('hub.html'),root/'index.html');(root/'hubdata.js').write_text('window.R52HUB='+json.dumps(fixtures,ensure_ascii=False)+';',encoding='utf8')
