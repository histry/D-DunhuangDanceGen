"""Prepare method-blind C1/C2 playback; this creates no human review records."""
import json,base64
from pathlib import Path
import numpy as np
from dunhuang.research.execution_planning_r58.common import ROOT,CFG,fresh,read,write,sha
from dunhuang.research.strong_baseline_r52.view import finish,encode
def main():
 out=fresh(ROOT/'review');d=read(ROOT/'controls/result.json');z=np.load(ROOT/'controls/arrays.npz');folder=fresh(out/'viewer');rng=np.random.default_rng(CFG['seed']);cases=[];mapping=[]
 for song in CFG['songs']:
  order=list(rng.permutation(['C1','C2']));records={r['arm']:r for r in d['records'] if r['song']==song};N=records['C1']['frames'];assert all(records[a]['status']=='NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW' for a in order);audio=Path(d['binding']['parent']['music'][song]['path']);assert sha(audio)==d['music'][str(song)]['audio_sha256'];methods=[]
  for k,arm in enumerate(order):
   P=z[f'song{song}_{arm}_P'];jerk=np.r_[np.zeros(3),np.sqrt(np.mean((np.diff(P.astype(float),n=3,axis=0)*27000)**2,axis=(1,2)))];methods.append(dict(name=['A','B'][k],data=encode(P),jerk=jerk.tolist()))
  cases.append(dict(label='盲审音乐'+str(song+1),frames=N,origin=0,audio='data:audio/wav;base64,'+base64.b64encode(audio.read_bytes()).decode(),methods=methods,meta=dict(scope='同一音乐、相机、尺度及播放时钟；允许差不多或无法判断。这里只准备媒体，没有人类评分。身体22关节Root-relative，世界Root和接触未知。',regions=[])));mapping.append(dict(blind_item_id='r58-'+str(song),A=order[0],B=order[1],audio_sha256=sha(audio)))
 finish(folder,cases,z['parents']);p=folder/'index.html';s=p.read_text(encoding='utf8').replace('R52','R58 盲审准备').replace('保留 Heuristic 强基线；生成比较同一完整来源与几何基线，修复评价时间坐标。来源阶段为观测运动学代理，世界 Root／接触及 Native24 尚未取得完整资格。','只评价可感知的自然流畅、音乐协调和差异；允许A/B/差不多/无法判断。不展示方法名称、来源身份、分数或数据角色。');p.write_bytes(s.encode());write(folder/'protocol.json',dict(files=[dict(name=q.name,bytes=q.stat().st_size,sha256=sha(q)) for q in sorted(folder.iterdir()) if q.name!='protocol.json']))
 write(out/'admin_mapping.json',dict(private_admin_only=True,items=mapping));write(out/'protocol.json',dict(state='PREPARATION_ONLY',human_reviewed=False,independent_records=0,media_source_sha256=sha(ROOT/'controls/arrays.npz'),files=read(folder/'protocol.json')['files'],questions=['naturalness_and_fluency','music_coordination','perceptible_difference'],choices=['A','B','SAME','UNSURE'],two_independent_reviewers_then_adjudication=True,required_fields=['reviewer_id','reviewer_roster_ref','reviewed_at','blind_item_id','media_sha256','decision','UNSURE_reason','prior_exposure'],music_boundary_task='SEPARATE_AUDIO_ONLY; NO_DANCE_OR_MODEL_PREDICTION; HUMAN_LABELS_NOT_CREATED',new_independent_music=False,native24='NOT_EVALUABLE'))
if __name__=='__main__':main()
