"""Read-only automatic audit of the frozen thirty execution positions."""
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
from .common import ROOT,HERE,CFG,PARENT,read,write,sha,uid,prepare,checked_parent,fresh,seal
from . import music
from .reviews import classification
from .reference import check as reference_check
from dunhuang.research.paper1_support import study as old
from dunhuang.research.warp_coupling_r53.common import array_hash,save_arrays
from dunhuang.research.complete_r36.core import relative_fk,rotation_sample
from dunhuang.research.gtw_kinematic_baseline.features import ReferenceCost

def dynamics(R,ctx):
    sig,P=old.metrics.signals(R,ctx['J'],ctx['parents']);d=old.metrics.describe(R,P,sig['body'],ctx['cal']['thresholds']['0.2']['body'])
    d={k:v for k,v in d.items() if k not in ['cell_angular','cell_linear','low_cells']}
    for n in [1,2,3]:d['FK_D%d_q95'%n]=float(np.quantile(np.linalg.norm(np.diff(P,n=n,axis=0)*30**n,axis=-1).mean(-1),.95))
    delta=R[:-1].swapaxes(-1,-2)@R[1:];w=Rotation.from_matrix(delta.reshape(-1,3,3)).as_rotvec().reshape(-1,22,3)*30
    alpha=((delta[:-1]@w[1:,...,None])[...,0]-w[:-1])*30
    d['SO3_angular_speed_q95']=float(np.quantile(np.linalg.norm(w,axis=-1).mean(-1),.95));d['SO3_transported_acceleration_q95']=float(np.quantile(np.linalg.norm(alpha,axis=-1).mean(-1),.95))
    V=np.diff(P,axis=0)*30;V-=V.mean(0);power=np.abs(np.fft.rfft(V,axis=0))**2;freq=np.fft.rfftfreq(len(V),1/30);total=float(power.sum())
    d['FK_velocity_high_frequency_fraction_above6Hz']=float(power[freq>6].sum()/total) if total>1e-24 else None
    return d,P

def source_time_preflight(ctx,cases):
    out=[]
    for c in cases:
        u=c['unit'];s=u['source_uid'];a,b=u['source_interval'];T=c['target_frames'];mask=ctx['natural'][s]['regions']['body']['low_cells'];found=[]
        for v in sorted(ctx['units'],key=lambda v:(v['source_uid'],v['source_interval'],v['unit_uid'])):
            x,y=v['source_interval']
            if v['source_uid']!=s or v['unit_uid']==u['unit_uid'] or abs((y-x)/(b-a)-1)>.1 or not .75<=(T-1)/(y-x-1)<=1.33:continue
            for h in ctx['natural'][s]['regions']['body']['low_runs']:
                l,r=h['cells_half_open']
                if x+4<l<r<y-5 and sum(not z for z in mask[x:l])>=30 and sum(not z for z in mask[r:y-1])>=30:found.append((v,h))
        counts=[]
        for unit,hold in [(u,c['hold'])]+(found[:1]):
            bank={}
            for w in old.clocks(unit,hold,T):
                q=np.interp(np.arange(T),w[:,1],w[:,0]);base=np.linspace(unit['source_interval'][0],unit['source_interval'][1]-1,T);q[:4]=base[:4];q[-4:]=base[-4:]
                if not old.duplicate_clock(q,bank):bank[array_hash(q)]=(q,None)
            counts.append(dict(event_uid=unit['unit_uid'],source_interval=unit['source_interval'],distinct_clocks=len(bank)))
        out.append(dict(case_key=c['key'],source_uid=s,status='REGISTERED_NONDEGENERATE_PAIR' if len(counts)==2 and min(v['distinct_clocks'] for v in counts)>1 else 'MISSING_NONDEGENERATE_PAIR',sources=counts,GTW_or_G1_used_to_select_source=False,new_G1_calls=0,execution='NOT_STARTED'))
    return out

def representation_audit():
    base=old.ROOT/'assets/change_official_14';m=read(base/'sources.json');rows=[]
    for s in m['sources']:
        p=base/s['file'];assert sha(p)==s['sha256']
        with np.load(p,allow_pickle=False) as z:
            assert z['poses'].shape==(s['frames'],165) and z['trans'].shape==(s['frames'],3) and np.isfinite(z['poses']).all() and np.isfinite(z['trans']).all()
        rows.append(dict(source_uid=s['source_id'],poses55_observed=True,source_translation_observed=True,declared_axis=m['coordinate_system'],units=m['translation_units'],SMPL24_hand_joint_mapping='UNVERIFIED',critic_coordinate_transform='UNVERIFIED',critic_status='NOT_EVALUABLE'))
    return dict(rows=rows,critic_preprocessed_input='B,T,25,3: 24_SMPL_AXIS_ANGLE_PLUS_ROOT',critic_original_demo_input='B,25,6,T',generated_body22_world_root=False,generated_missing_SMPL_hand_rotations=[22,23],G1_root_contract='UNOBSERVED',zero_padding_performed=False,weights_downloaded=False,critic_inference_calls=0,status='NOT_EVALUABLE',reason='SMPLX55_TO_SMPL24_HANDS_AND_CRITIC_COORDINATES_NOT_VERIFIED; GENERATED_G1_WORLD_ROOT_ABSENT')

def preview(dest=None):
    out=fresh(dest or ROOT/'preview');reg,records=checked_parent();ctx=prepare();cats=music.catalogs();z=np.load(PARENT/'arrays.npz',allow_pickle=False);cases={c['key']:c for c in reg['cases']};by={(r['case_key'],r['method']):r for r in records};profiles={};rows=[];bindings={};scorers={}
    write(out/'registration.json',dict(protocol=CFG,parent_manifest=CFG['q01_controls_manifest'],parent_input_hashes=reg['input_files'],source_time_preflight=source_time_preflight(ctx,reg['cases']),music_role='READ_ONLY_NO_SOURCE_OR_CLOCK_SELECTION',population='SIX_HISTORICALLY_EXPOSED_SOURCES_AND_THREE_ASSIGNED_SONGS_NOT_PAIRED_GROUND_TRUTH'))
    for r in records:
        c=cases[r['case_key']];n=c['key']+'__'+r['method'];R=z[n+'__R'];q=z[n+'__q'];h=array_hash(R);s=r['source_uid'];a,b=r['source_interval'];A=c['music_start_frame'];T=c['target_frames'];song=c['music_song'];binding=music.bind(cats[song],song,[A,A+2*T+48]);bindings[binding['id']]=binding
        assert h==r['audit']['rotation_sha256'] and array_hash(rotation_sample(ctx['lib'].rotations[s],q))==h
        pid=uid([c['key'],s,a,b,h,binding['id']])
        if pid not in profiles:
            d,P=dynamics(R,ctx);full,connection=old.connect(ctx,R,c['neighbor'],T)
            assert connection['passes_original_screen']==r['audit']['bridge_screen']
            assert (None if full is None else array_hash(full))==r['full_rotation_sha256']
            sk=(s,a,b)
            if sk not in scorers:scorers[sk]=ReferenceCost(ctx['lib'].rotations[s][max(0,a-30):min(len(ctx['lib'].rotations[s]),b+30)],ctx['J'],ctx['parents'])
            cost=scorers[sk].score(R,only='GTWLocal')[0]['GTWLocal'];assert abs(cost-r['GTWLocal'])<=1e-12
            beats=music.motion_beats(P)+A/30;events=music.motion_events(R)
            profiles[pid]=dict(id=pid,rotation_sha256=h,dynamics=d,GTWLocal=cost,motion_minima_seconds=beats.tolist(),automatic_motion_events=events,music_scores=music.scores(beats,cats[song],cats[(song+1)%3],A/30,(A+T-1)/30),music_evidence_ref=binding['id'],clock_interpretation='ASSIGNED_MUSIC_NOT_CHANG_E_RECORDED_PAIR',G1_recomputed=True)
        rows.append(dict(case_key=c['key'],method=r['method'],source_uid=s,source_interval=[a,b],profile_ref=pid,rotation_sha256=h,clock_sha256=array_hash(q),music_evidence_ref=binding['id'],G1_pass=r['audit']['bridge_screen'],G1_frozen_metrics=r['connection']['metrics'],source_resampling_exact=True,mapped_source_low_seconds=r['mapped_low_seconds'],MotionCritic=None,MotionCritic_status='NOT_EVALUABLE',quality='NOT_EVALUABLE'))
    comparisons=[]
    for c in reg['cases']:
        for left,right,contrast in [('original_GTWLocal','original_Uniform','TIMING_ONLY'),('alternative_Uniform','original_Uniform','SOURCE_ONLY_UNIFORM'),('alternative_GTWLocal','alternative_Uniform','ALTERNATIVE_TIMING_ONLY')]:
            L=by.get((c['key'],left));R=by.get((c['key'],right));get=lambda r,m:None if r is None or not r['audit']['bridge_screen'] or c['key']+'__'+m+'__full_R' not in z else z[c['key']+'__'+m+'__full_R'];status=classification(get(L,left),get(R,right))
            deltas=None
            if status=='VALID_DISTINCT':
                ref=lambda m:next(v['profile_ref'] for v in rows if v['case_key']==c['key'] and v['method']==m)
                lp=profiles[ref(left)];rp=profiles[ref(right)];primary=lambda p:next(v['score'] for v in p['music_scores'] if v['stream']=='beat_this:beat' and v['condition']=='assigned');x,y=primary(lp),primary(rp)
                deltas=dict(BAS3D22=None if x is None or y is None else x-y,GTW_reference_cost=lp['GTWLocal']-rp['GTWLocal'],FK_D2_q95=lp['dynamics']['FK_D2_q95']-rp['dynamics']['FK_D2_q95'],FK_D3_q95=lp['dynamics']['FK_D3_q95']-rp['dynamics']['FK_D3_q95'],SO3_transported_acceleration=lp['dynamics']['SO3_transported_acceleration_q95']-rp['dynamics']['SO3_transported_acceleration_q95'],longest_low_seconds=lp['dynamics']['longest_low_seconds']-rp['dynamics']['longest_low_seconds'],mapped_source_low_seconds=L['mapped_low_seconds']-R['mapped_low_seconds'],MotionCritic=None,quality_winner=None)
            comparisons.append(dict(case_key=c['key'],source_uid=c['unit']['source_uid'],contrast=contrast,left=left,right=right,classification=status,deltas=deltas))
    write(out/'executions.json',rows);write(out/'profiles.json',list(profiles.values()));write(out/'music_bindings.json',list(bindings.values()));write(out/'comparisons.json',comparisons);write(out/'representation_audit.json',representation_audit());write(out/'reference_parity.json',reference_check())
    write(out/'summary.json',dict(execution_positions=len(rows),unique_core_profiles=len(profiles),classifications=dict(Counter(r['classification'] for r in comparisons)),G1_passes=sum(r['G1_pass'] for r in rows),human_records=0,music_selection_used=False,quality='NOT_EVALUABLE',BAP='NOT_EVALUABLE_NO_REGISTERED_MUSIC_TARGET_ANCHORS',MotionCritic='NOT_EVALUABLE',PFC_FootSliding_Native24='NOT_EVALUABLE',joint_execution='NOT_STARTED',training_steps=0,can_route=False));z.close();seal(out);print('AUTO_EVAL_PREVIEW_READY',len(rows),len(profiles),flush=True)

if __name__=='__main__':preview()
