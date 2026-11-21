"""Apply the frozen final model to actual old candidate rotations; no new search."""
import sys
import numpy as np
import torch
from .common import *
from .train import load
from .network import tensors
from dunhuang.research.role_alignment_q01.common import ROOT as ROLE,verify as verify_role,prepare
from dunhuang.research.transition_r37.content import observations
from dunhuang.research.complete_r36.core import relative_fk

def raw_music(song):
 import sys
 from dunhuang.research.song_transfer.music import runtime
 runtime(read('inputs/clock_assets_v2.json'));legacy=Path('third_party/danceRAG').resolve();sys.path.insert(0,str(legacy))
 from evaluation.paired_alignment_observations_v3 import full_audio_observations
 p=Path('outputs/complete_r36_01/viewer/audio')/('dunhuangwu%d.wav'%(song+2));M,MM,proof=full_audio_observations(str(p));return M,MM,dict(audio_sha256=sha(p),extractor=proof)

def run(out=None,names=None):
 names=names or ['Feasible'];out=fresh(out or ROOT/'preview');verify_role(ROLE/'preview');verify_role(ROLE/'controls');reg=read(ROLE/'preview/registration.json');candidates=read(ROLE/'preview/candidates.json');normal=dict(np.load(ROOT/'data/normalizer.npz'));ctx=prepare();nets={n:load(n) for n in names};music={};proof={};tracks={};items=[];selections=[]
 with np.load(ROLE/'preview/arrays.npz',allow_pickle=False) as z:
  def track(key):
   if key not in tracks:tracks[key]=np.round(relative_fk(z[key],ctx['J'],ctx['parents']),4).tolist()
   return key
  for c in reg['cases']:
   if c['requested_nodes']!=3:continue
   own=[r for r in candidates if r['case_key']==c['key'] and r['G1_pass']];T=c['target_frames'];A=c['music_start_frame'];song=c['music_song']
   if song not in music:music[song],proof[song]=None,None;M,MM,p=raw_music(song);music[song]=(M,MM);proof[song]=p
   M,MM=music[song];M=np.where(MM[A:A+T],(M[A:A+T]-normal['M_mean'])/normal['M_std'],0).astype(np.float32);MV=MM[A:A+T].any(-1);clips=[]
   for r in own:
    D,DM=observations(z[r['uid']+'__R'],ctx['J'],ctx['parents']);DM[0,198:272]=False;DM[:,-2:]=False;D=np.where(DM,(D-normal['D_mean'])/normal['D_std'],0).astype(np.float32);clips.append((M,D,DM,MV))
   variants=[];base=next(r for r in own if 'E1W' in r['labels']);chosen={'E1-W':base}
   for name,net in nets.items():
    with torch.inference_mode():scores=net(*tensors(clips),constrained=True,uniform_only=True).cpu().numpy()
    selected=min(range(len(own)),key=lambda i:(-float(scores[i]),own[i]['GTWLocal'],own[i]['uid']));chosen[name]=own[selected];selections.append(dict(case_key=c['key'],source_uid=c['source_uid'],model=name,selected_uid=own[selected]['uid'],scores=[dict(uid=r['uid'],score=float(v)) for r,v in zip(own,scores)],candidate_budget=len(own),selection_status='SHADOW_UNQUALIFIED_FINAL_MODEL',G1_remeasured_in_parent=True,actual_bridge_sha256=own[selected]['connection']['bridge_sha256'],can_route=False))
   for name,r in chosen.items():variants.append(dict(label=name+' · Shadow · G1 '+str(r['G1_pass']),track=track(r['uid']+'__full_R'),qualified=True,source=str(c['source_interval']),q=z[r['uid']+'__q'].tolist(),source_interval=c['source_interval'],details=dict(selected_uid=r['uid'],G1_pass=r['G1_pass'],independent_artistic_quality='NOT_EVALUABLE')))
   audio='music_'+str(song)+'_'+str(T)+'.wav';items.append(dict(label=c['source_uid']+' · 实际模型选择',frames=2*T+48,core=T,target=None,raw=False,audio=audio,variants=variants));key=c['key']+'__raw_R';items.append(dict(label=c['source_uid']+' · 原速来源',frames=len(z[key]),core=0,target=None,raw=True,audio=None,variants=[dict(label='原速来源',track=track(key),qualified=None)]))
 template=Path('src/dunhuang/research/source_time_q01/viewer.html').read_text(encoding='utf8').replace("c.fillRect(x.target/x.frames*300,325,2,28);","if(x.target!=null)c.fillRect(x.target/x.frames*300,325,2,28);")
 payload=dict(items=items,tracks=tracks,parents=ctx['parents'].tolist(),intro='FGMMC-Q01：真实训练后的有限合法对应模型。保持相同来源、Core时长、既有候选和原G1。当前仅Shadow展示，不表示音乐对应或艺术质量已获资格；原速来源独立入口。');viewer=fresh(out/'viewer');(viewer/'index.html').write_bytes(template.replace('__DATA__',__import__('json').dumps(payload,ensure_ascii=False,separators=(',',':'))).encode());audio=[]
 for song,(M,MM) in music.items():
  T=next(c['target_frames'] for c in reg['cases'] if c['music_song']==song);name='music_'+str(song)+'_'+str(T)+'.wav';p=Path('outputs/paper1_auto_eval_01/preview/viewer')/name;audio.append(dict(name=name,existing_relative_path=str(p),sha256=sha(p)))
 write(viewer/'protocol.json',dict(files={'index.html':dict(bytes=(viewer/'index.html').stat().st_size,sha256=sha(viewer/'index.html'))},existing_local_audio=audio,models=names,can_route=False));write(out/'selections.json',selections);write(out/'music_proof.json',proof);seal(out);print('FG_PREVIEW_READY',len(items),flush=True)
if __name__=='__main__':run(ROOT/'preview' if sys.argv[-1]!='controls' else ROOT/'controls_view',None if sys.argv[-1]!='controls' else CFG['models'])
