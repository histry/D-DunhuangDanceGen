"""Compile a frozen Heuristic plan, full source memory and genuine target music."""
from pathlib import Path
import numpy as np
import soundfile as sf
from .common import ROOT,read,write,sha,fresh
from .contracts import VERSION,validate
from dunhuang.research.transition_planning_r50.common import assets
from dunhuang.research.bridge_r38.study import runtime
from dunhuang.research.validation_r51.condition import install


def run(out):
 out=fresh(out);net,lib,units,tracks=assets();plans=read(ROOT/'plans/result.json');arrays=np.load(ROOT/'plans/arrays.npz');rt=runtime(read('src/dunhuang/research/bridge_r38/config.json'));fixtures=[]
 assert np.max(abs(net.joints-rt.s.J[0,:22].cpu().numpy()))<1e-6
 for song in (0,1):
  row=next(r for r in plans['records'] if r['song']==song and r['events']==2 and r['gap']==48);name='fixture'+str(song);selection=next(iter(row['selected'].values()));key=row['key']+'_'+next(iter(row['selected']));R=arrays[key+'_R'];a,b=row['music_interval'];source=R.copy();valid=np.ones((len(R),22),bool);ids=np.full(len(R),-1,np.int64);positions=np.full(len(R),np.nan);events=[]
  for index,(e,span) in enumerate(zip(selection['events'],[s['interval'] for s in row['regions'] if s['kind']=='core'])):
   u,v=span;lo,hi=u-a,v-a;ids[lo:hi]=index;w=np.array(e['warp']['time_warp']);positions[lo:hi]=np.interp(np.arange(u,v),w[:,1],w[:,0])
   events.append(dict(event_uid=e['event_uid'],source_uid=e['source_uid'],source_interval=e['source_interval'],target_interval=[u,v],time_warp=e['warp']['time_warp'],phase_scope='KINEMATIC_PROXY_NOT_SEMANTIC_TRUTH',phase=dict(status='SEMANTIC_UNKNOWN',kinematic_candidates=next(u for u in units if u['unit_uid']==e['event_uid'])['phases']),phase_valid=False,pose_anchors=[dict(source_frame=float(s),target_frame=float(t),status='REFERENCE_SOFT') for s,t in w],left_context=dict(kind='REFERENCE_EVENT' if index else 'UNKNOWN',event_uid=selection['events'][index-1]['event_uid'] if index else None),right_context=dict(kind='REFERENCE_EVENT' if index+1<len(selection['events']) else 'UNKNOWN',event_uid=selection['events'][index+1]['event_uid'] if index+1<len(selection['events']) else None),transition_budget=48 if index+1<len(selection['events']) else 0,source_asset=lib.rows[e['source_uid']][0]['reference'],sdmr_prior=dict(qualified=False,consumed_by_denoiser=False,score=None)))
  for span in row['regions']:
   if span['kind']=='transition':u,v=span['interval'];valid[u-a:v-a]=False;source[u-a:v-a]=np.nan
  wave,sr=sf.read(tracks[song]['path'],always_2d=True,dtype='float32');clip=wave[round(a*sr/30):round(b*sr/30)];wav=out/(name+'.wav');sf.write(wav,clip,sr,subtype='FLOAT')
  feature,_=rt.e(str(wav),skip_completed=False);feature=np.asarray(feature[:len(R)],np.float32);assert feature.shape==(len(R),35) and np.isfinite(feature).all()
  plan=validate(dict(schema=VERSION,scope='FROZEN_HEURISTIC_REFERENCE_NOT_SDMR_SUCCESS',fps=30,target_interval=[a,b],coordinate_uid='R31_COMMON22_ROW6D_BODY_RELATIVE',root_truth='UNKNOWN',contact_truth='UNKNOWN',events=events,regions=row['regions'],music=dict(audio=dict(path=str(wav),sha256=sha(wav)),features35_key='music35',feature_clock='30HZ_CROP_CENTERED_STFT_FINAL_EXTRA_FRAME_DISCARDED',features35_source='PINNED_OFFICIAL_LODGE_EXTRACTOR'),source_sampling='SO3_INTERPOLATED_WITH_OBSERVED_NATIVE_HALO; SOURCE_INTERVAL_REMAINS_COMPLETE',native24='NOT_EVALUABLE_22_OF24'))
  path=out/(name+'.npz');np.savez_compressed(path,anchor_R=R,reference_R=source,reference_valid=valid,event_ids=ids,source_positions=positions,music35=feature,J=net.joints,parents=net.parents)
  plan['compiled']=dict(path=str(path),sha256=sha(path));write(out/(name+'.json'),plan);fixtures.append(dict(key=name,plan=str(out/(name+'.json')),plan_sha256=sha(out/(name+'.json'))));print('R52_COMPILE',name,len(R),int(valid.any(-1).sum()),flush=True)
 rt.c.context_branch.zero_grad(set_to_none=True);rt.check();rt.close();write(out/'result.json',dict(fixtures=fixtures,interface=VERSION,root_contact='UNKNOWN',training_scope='SOURCE_RECONSTRUCTION_AND_TRUE_CONTINUOUS_INBETWEENING_SEPARATE',sdmr_training_gate_on_moduleII=False))
