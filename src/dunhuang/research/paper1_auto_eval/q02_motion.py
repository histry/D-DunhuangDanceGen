"""Prospective orientation and complete write-stencil diagnostics, not an art judge."""
import numpy as np
from scipy.spatial.transform import Rotation
from .q02_common import CFG,prepare,checked_parent,read
from .experiment import dynamics
from .observability import angular_acc
from . import controls,music
from dunhuang.research.gtw_kinematic_baseline.features import ReferenceCost
from dunhuang.research.complete_r36.core import relative_fk
from dunhuang.research.warp_coupling_r53.common import array_hash

def variants(source,c,beats):
    a,b=c['unit']['source_interval'];base=source[a:b];T=len(base);mask=np.zeros(T,bool);mask[4:-4]=True
    for label,R,q,m,scope in controls.perturbations(source,c):
        if label in ['SourceOriginal','Identical','LegalHoldExtension','LegalRateCorner','TimeSkip','PoseRepeat','JointJump']:
            yield label,R,q,m,scope,None
    yield 'Static',np.repeat(base[:1],T,axis=0),None,np.ones(T,bool),'SYNTHETIC_ALL_STATIC_NOT_ARTISTIC_HOLD_TRUTH',None
    if len(beats)<2 or np.median(np.diff(beats))<=0:
        # Keep registered missing controls, never select a convenient frequency.
        for dose in CFG['doses_rad']:yield 'BeatOscillation_'+str(dose),None,None,mask,'MISSING_ASSIGNED_BEAT_PERIOD',dose
    for dose in CFG['doses_rad']:
        d=np.zeros((T,3));d[:,0]=np.where(np.arange(T)%2,1.,-1.)*dose;d[~mask]=0;R=base.copy();R[:,[20,21]]=R[:,[20,21]]@Rotation.from_rotvec(d).as_matrix()[:,None];R[~mask]=base[~mask]
        yield 'WristNoise_'+str(dose),R,None,mask.copy(),'SYNTHETIC_LOCAL_ROTATION_NOISE',dose
        if len(beats)<2:continue
        dt=float(np.median(np.diff(beats)));t=np.arange(T)/30;A=c['music_start_frame']/30;phase=beats[0]-A
        # Half beat frequency: absolute sinusoid velocity has two minima per cycle.
        edge=np.minimum(np.arange(T)-3,T-4-np.arange(T))/max(1,30*CFG['envelope_seconds']);env=np.clip(edge,0,1);env=env*env*(3-2*env)
        d[:,0]=dose*env*np.sin(np.pi*(t-phase)/dt);d[~mask]=0;R=base.copy();R[:,[18,19]]=R[:,[18,19]]@Rotation.from_rotvec(d).as_matrix()[:,None];R[~mask]=base[~mask]
        yield 'BeatOscillation_'+str(dose),R,None,mask.copy(),'SYNTHETIC_SMALL_BEAT_PERIOD_OSCILLATION_NOT_SOURCE_WARP',dose

def build_arrays():
    ctx=prepare();reg,_=checked_parent();cats=music.catalogs();rows=[];arrays={}
    for c in reg['cases']:
        s=c['unit']['source_uid'];a,b=c['unit']['source_interval'];A=c['music_start_frame']/30;cat=cats[c['music_song']]
        beats=np.asarray([e['seconds'] for e in cat['events'] if e['source']=='beat_this' and e['kind']=='beat' and e['seconds_supported'] and A<=e['seconds']<A+(b-a)/30])
        for label,R,q,mask,scope,dose in variants(ctx['lib'].rotations[s],c,beats):
            n=c['key']+'__'+label;row=dict(key=n,case_key=c['key'],source_uid=s,source_interval=[a,b],label=label,scope=scope,dose_rad=dose,write_frames=np.flatnonzero(mask).tolist(),assigned_music_song=c['music_song'],music_start_frame=c['music_start_frame'],frames=b-a,beat_period_seconds=None if len(beats)<2 else float(np.median(np.diff(beats))),status='MISSING' if R is None else 'OBSERVED',artistic_quality='NOT_EVALUABLE')
            if R is not None:
                base=ctx['lib'].rotations[s][a:b];assert np.isfinite(R).all() and np.max(abs(R.swapaxes(-1,-2)@R-np.eye(3)))<1e-6
                assert np.array_equal(R[~mask],base[~mask]);row['outside_write_exact']=True;row['rotation_sha256']=array_hash(R);arrays[n+'__R']=R
                if q is not None:arrays[n+'__q']=q
            rows.append(row)
        print('Q02_PREVIEW_SOURCE',s,flush=True)
    return rows,arrays

def score(rows,arrays):
    ctx=prepare();cats=music.catalogs();results=[];models={};baselines={}
    for r in rows:
        if r['status']!='OBSERVED':results.append(r);continue
        R=arrays[r['key']+'__R'];a,b=r['source_interval'];source=ctx['lib'].rotations[r['source_uid']];base=source[a:b];d,P=dynamics(R,ctx);Pb=relative_fk(base,ctx['J'],ctx['parents']);acc=angular_acc(R);ab=angular_acc(base)
        mask=np.zeros(len(R),bool);mask[r['write_frames']]=True;m4=np.lib.stride_tricks.sliding_window_view(mask,4).any(1);m3=np.lib.stride_tricks.sliding_window_view(mask,3).any(1)
        D=np.linalg.norm(np.diff(P,n=3,axis=0)*30**3,axis=-1);Db=np.linalg.norm(np.diff(Pb,n=3,axis=0)*30**3,axis=-1)
        mx=lambda v,m:None if not m.any() else float(v[m].max())
        if r['source_uid'] not in models:models[r['source_uid']]=ReferenceCost(source[max(0,a-30):min(len(source),b+30)],ctx['J'],ctx['parents'])
        A=r['music_start_frame']/30;end=A+(len(R)-1)/30;cat=cats[r['assigned_music_song']];beats=music.motion_beats(P)+A;query=np.asarray([e['seconds'] for e in cat['events'] if e['source']=='beat_this' and e['kind']=='beat' and e['seconds_supported'] and A+.5<=e['seconds']<end-.5])
        wrist=Rotation.from_matrix((base[:,[20,21]].swapaxes(-1,-2)@R[:,[20,21]]).reshape(-1,3,3)).as_rotvec().reshape(-1,2,3)
        results.append(dict(r,dynamics=d,GTWLocal=models[r['source_uid']].score(R,only='GTWLocal')[0]['GTWLocal'],BAS=music.gaussian_match(query,beats),music_event_count=len(query),motion_minima_count=len(beats),motion_minima_hz=float(len(beats)/((len(R)-1)/30)),local_FK_D3_max=mx(D,m4),baseline_same_stencil_FK_D3_max=mx(Db,m4),local_FK_D3_max_abs_change=mx(abs(D-Db),m4),local_SO3_acc_max=mx(acc,m3),baseline_same_stencil_SO3_acc_max=mx(ab,m3),local_SO3_acc_max_abs_change=mx(abs(acc-ab),m3),FK_position_max_abs_change=float(abs(P-Pb).max()),wrist_local_max_degrees=float(np.linalg.norm(wrist,axis=-1).max()*180/np.pi),BAS_empty='NOT_EVALUABLE' if not len(beats) or not len(query) else None))
    deltas=[]
    for r in results:
        if r['status']!='OBSERVED':continue
        base=next(v for v in results if v['case_key']==r['case_key'] and v['label']=='SourceOriginal')
        deltas.append(dict(key=r['key'],label=r['label'],source_uid=r['source_uid'],dose_rad=r['dose_rad'],BAS_delta=None if r['BAS'] is None or base['BAS'] is None else r['BAS']-base['BAS'],FK_D3_delta=r['dynamics']['FK_D3_q95']-base['dynamics']['FK_D3_q95'],SO3_acc_delta=r['dynamics']['SO3_transported_acceleration_q95']-base['dynamics']['SO3_transported_acceleration_q95'],local_FK_change=r['local_FK_D3_max_abs_change'],local_SO3_change=r['local_SO3_acc_max_abs_change'],low_seconds_delta=r['dynamics']['longest_low_seconds']-base['dynamics']['longest_low_seconds'],GTWLocal_delta=r['GTWLocal']-base['GTWLocal'],music_winner=None))
    return results,deltas
