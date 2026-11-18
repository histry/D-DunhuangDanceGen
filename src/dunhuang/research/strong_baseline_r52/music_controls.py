"""Frozen-choice acoustic counterfactuals; all alignment labels are proxies."""
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks
from .common import ROOT,read,write,fresh,relative_fk
from .view import encode,finish
from dunhuang.research.transition_planning_r50.common import assets
from dunhuang.research.transition_planning_r50.planning import bridge
from dunhuang.research.complete_r36.core import rotation_sample
from dunhuang.research.validation_r51.common import regions


def proxies(P,beats,energy):
 speed=np.r_[0,np.sqrt(np.mean((np.diff(P,axis=0)*30)**2,axis=(1,2)))];peaks,_=find_peaks(speed,distance=8,prominence=max(float(np.ptp(speed))*.1,1e-8))
 beats=np.asarray(beats);dist=np.min(abs(peaks[:,None]-beats[None]),axis=1) if len(peaks) and len(beats) else np.array([])
 correlations={}
 for width in (15,30,60):
  x=np.convolve(speed,np.ones(width)/width,'same');y=np.convolve(energy,np.ones(width)/width,'same')
  correlations[str(width)]=float(np.corrcoef(x,y)[0,1]) if min(x.std(),y.std())>1e-10 else None
 return dict(peak_count=len(peaks),nearest_beat_frames=dist.tolist(),peak_to_beat_MAE_frames=float(dist.mean()) if len(dist) else None,beat_alignment_proxy=float(np.exp(-dist**2/18).mean()) if len(dist) else None,activity_energy_correlations=correlations,label_scope='AUTOMATIC_KINEMATIC_ACOUSTIC_PROXY_NOT_EVENT_TRUTH')


def run(out):
 out=fresh(out);viewer=out/'viewer';viewer.mkdir();net,lib,units,tracks=assets();records=[];cases=[];arrays={};z=np.load(ROOT/'plans/arrays.npz');ref=read('outputs/bridge_dynamics_r48_01/main/result.json')['reference']
 for r in read(ROOT/'plans/result.json')['records']:
  if r['events']!=2 or r['gap']!=48:continue
  selection=next(iter(r['selected'].values()));a,b=r['music_interval'];R=z[r['key']+'_'+next(iter(r['selected']))+'_R'];chunks=[]
  for e in selection['events']:
   s,t=e['source_interval'];chunks.append(rotation_sample(lib.rotations[e['source_uid']],np.linspace(s,t-1,240)))
  q,evidence=bridge(chunks[0],chunks[1],48,net,ref);linear=None if q is None else np.concatenate([chunks[0],q,chunks[1]])
  definitions=[('CorrectE1W',r['song'],a,R),('WrongSong',(r['song']+1)%3,a,R),('SameSongWrongTime',r['song'],a+240,R)]
  if linear is not None:definitions.append(('LinearNoBeat',r['song'],a,linear))
  arms={};review_items=[]
  for index,(name,song,start,rot) in enumerate(definitions):
   P=relative_fk(rot,net.joints,net.parents);end=start+len(P);track=tracks[song];assert end<=len(track['signal']['energy']);beats=np.asarray(track['structure']['beats']);beats=beats[(beats>=start)&(beats<end)]-start
   audio=r['key']+'_'+name+'.wav'
   with sf.SoundFile(track['path']) as f:
    f.seek(round(start*f.samplerate/30));wave=f.read(round(end*f.samplerate/30)-round(start*f.samplerate/30));sr=f.samplerate
   sf.write(viewer/audio,wave,sr);arrays[r['key']+'_'+name]=P
   arms[name]=dict(audio_song=song,audio_interval=[start,end],source_events=selection['events'],metrics=proxies(P,beats,np.asarray(track['signal']['energy'][start:end])),kinematics=regions(rot,net.joints,net.parents,r['regions'],origin=a),warp='LINEAR_ENDPOINTS_NO_BEAT' if name=='LinearNoBeat' else 'FROZEN_E1W')
   # Separate anonymous items retain original music clocks; do not falsely sync different audio.
   blind='R52-'+str(r['song'])+'-'+str([2,4,1,3][index]);review_items.append(dict(blind_id=blind,method=name))
   cases.append(dict(label=blind,frames=len(P),origin=start,audio=audio,methods=[dict(name='匿名动作',data=encode(P),jerk=np.r_[np.zeros(3),np.sqrt(np.mean((np.diff(P,n=3,axis=0)*27000)**2,axis=(1,2)))].tolist())],meta=dict(scope='匿名观感准备：自然流畅／音乐协调／可感知差异；允许差不多或无法判断。未开展人工评审。',root_contact='UNKNOWN')))
  records.append(dict(key=r['key'],arms=arms,no_beat_bridge=evidence,review_mapping=review_items))
 finish(viewer,sorted(cases,key=lambda c:c['label']),net.parents);np.savez_compressed(out/'arrays.npz',**arrays);write(out/'result.json',dict(records=records,human_reviewed=False,review_status='PREPARATION_ONLY',can_route=False,training_steps=0))
