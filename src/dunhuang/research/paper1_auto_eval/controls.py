"""Known clock and motion interventions; never manufacture human preferences."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .common import ROOT,CFG,read,write,sha,prepare,fresh,seal,checked_parent
from .experiment import dynamics
from . import music
from dunhuang.research.paper1_support import study as old
from dunhuang.research.complete_r36.core import rotation_sample
from dunhuang.research.warp_coupling_r53.common import array_hash,save_arrays
from dunhuang.research.gtw_kinematic_baseline.features import ReferenceCost

def perturbations(source,c):
    u=c['unit'];a,b=u['source_interval'];base=source[a:b];T=len(base);q0=np.arange(a,b,dtype=float);mask=np.zeros(T,bool);mask[4:-4]=True
    yield 'SourceOriginal',base.copy(),q0,np.zeros(T,bool),'OBSERVED_SOURCE'
    yield 'Identical',base.copy(),q0,np.zeros(T,bool),'IDENTITY_CONTROL'
    R=base.copy();sign=np.where(np.arange(T)%2,1.,-1.);d=np.zeros((T,3));d[:,0]=sign*CFG['jitter_wrist_rad'];d[~mask]=0
    R[:,[20,21]]=R[:,[20,21]]@Rotation.from_rotvec(d).as_matrix()[:,None];R[~mask]=base[~mask]
    yield 'WristJitter',R,None,mask.copy(),'SYNTHETIC_POSE_EDIT_NOT_SOURCE_PRESERVING'
    l,r=c['hold']['cells_half_open'];warps=old.clocks(u,c['hold'],T)
    best=min(warps,key=lambda w:(-float(np.diff(np.interp([l,r],w[:,0],w[:,1]))[0]),float(np.max(abs(w[:,1]-(w[:,0]-a))))))
    q=np.interp(np.arange(T),best[:,1],best[:,0]);q[:4]=q0[:4];q[-4:]=q0[-4:]
    yield 'LegalHoldExtension',rotation_sample(source,q),q,mask.copy(),'LEGAL_WARP_NOT_KNOWN_ARTISTIC_ERROR'
    middle=(a+b-1)/2;xs=[a,a+3,middle,b-4,b-1];ys=[0,3,3+.75*(T-7)/2,T-4,T-1];q=np.interp(np.arange(T),ys,xs);q[:4]=q0[:4];q[-4:]=q0[-4:]
    yield 'LegalRateCorner',rotation_sample(source,q),q,mask.copy(),'LEGAL_PIECEWISE_CLOCK_RATE_CHANGE_NOT_ARTISTIC_LABEL'
    q=q0.copy();mid=T//2;last=T-4;q[mid:last]+=np.linspace(6,0,last-mid);q[:4]=q0[:4];q[-4:]=q0[-4:];m=np.zeros(T,bool);m[mid:last]=True
    yield 'TimeSkip',rotation_sample(source,q),q,m,'ILLEGAL_LOCAL_CLOCK_RATE_CONTROL_NOT_A_CANDIDATE'
    R=base.copy();m=np.zeros(T,bool);m[mid-15:mid+15]=True;R[m]=base[mid]
    yield 'PoseRepeat',R,None,m,'SYNTHETIC_30_FRAME_REPEAT_WITH_ENTRY_EXIT_JUMPS_NOT_ARTISTIC_HOLD_TRUTH'
    R=base.copy();m=np.zeros(T,bool);m[mid:mid+CFG['joint_offset_frames']]=True;R[m,18]=R[m,18]@Rotation.from_rotvec([CFG['joint_offset_elbow_rad'],0,0]).as_matrix()
    yield 'JointJump',R,None,m,'SYNTHETIC_LOCAL_JOINT_ROTATION_NOT_CLINICAL_JOINT_LIMIT_TEST'

def paired():
    from dunhuang.research.timing_validity_r66 import population,engine
    from dunhuang.research.timing_validity_r66.common import CFG as legacy
    assert sha(legacy['data'])==legacy['data_sha256'];pop=population.select(read(legacy['data']),legacy)
    base=Path('outputs/music_events_01/paired');assert sha(base/'population.json')==CFG['paired_population_sha256'];reg=read(base/'population.json');cats={}
    for i,s in enumerate(reg['sources']):
        p=base/str(i)/'catalog.json';assert sha(p)==CFG['paired_catalog_sha256'][i];cat=read(p);assert cat['audio_sha256']==s['audio_sha256']
        for kind,values in (s.get('legacy_events') or {}).items():
            if not any(e['source']=='R64' and e['kind']==kind for e in cat['events']):cat['events'].extend(dict(source='R64',kind=kind,seconds=float(t),seconds_supported=0<=t<cat['duration_seconds']) for t in values)
        cats[s['source_uid']]=cat
    ctx=prepare();rows=[]
    for i,s in enumerate(reg['sources']):
        if s['status']!='SUPPORTED':rows.append(dict(source_uid=s['source_uid'],status=s['status']));continue
        raw=pop['sources'][s['source_uid']];assert raw['audio_sha256']==s['audio_sha256'];R,proof=engine.load_motion(raw)
        donor=next(v for v in reg['sources'][i+1:]+reg['sources'][:i] if v['dataset']==s['dataset'] and v['status']=='SUPPORTED' and v['audio_sha256']!=s['audio_sha256'])
        own,wrong=cats[s['source_uid']],cats[donor['source_uid']];lo,hi=proof['shared_source_interval'];offset=raw['audio_offset_frames']/30;start=max(0,lo/30+offset);end=min((hi-1)/30+offset,own['duration_seconds'],wrong['duration_seconds'])
        P=old.relative_fk(R,ctx['J'],ctx['parents']);beats=music.motion_beats(P)+offset;beats=beats[(beats>=start)&(beats<=end)]
        rows.append(dict(source_uid=s['source_uid'],dataset=s['dataset'],audio_group=s['audio_sha256'],donor_audio_group=donor['audio_sha256'],status='OBSERVED',clock_scope=s['clock_scope'],motion_sha256=array_hash(R),shared_seconds=[start,end],scores=music.scores(beats,own,wrong,start,end),source_support_proof=proof,exposure='HISTORICALLY_EXPOSED_TRAIN_NOT_NEW_TEST',quality='NOT_EVALUABLE'))
        print('AUTO_PAIRED_GROUP',s['source_uid'],flush=True)
    summary=[];rng=np.random.default_rng(76002)
    for dataset in ['AIST++','FineDance']:
        for stream in [a+':'+b for a,b in music.STREAMS]:
            values=[];own=[]
            for r in rows:
                if r.get('dataset')!=dataset or 'scores' not in r:continue
                ss={s['condition']:s['score'] for s in r['scores'] if s['stream']==stream};v=ss['assigned'];other=[ss['shift_%+.2f'%d] for d in CFG['clock_shifts_seconds'] if d]
                if v is not None and all(x is not None for x in other):values.append(float(np.mean([1 if v>x+1e-12 else .5 if abs(v-x)<=1e-12 else 0 for x in other])))
                own.append(dict(group=r['audio_group'],scores=ss))
            ci=None if len(values)<2 else np.quantile(rng.choice(values,(2000,len(values)),replace=True).mean(1),[.025,.975]).tolist()
            summary.append(dict(dataset=dataset,stream=stream,groups=len(values),correct_vs_shift_win=None if not values else float(np.mean(values)),group_bootstrap_95=ci,group_details=own))
    return rows,summary

def run(dest=None):
    if dest is None:
        receipt=read(ROOT/'receipts/preview_browser_acceptance.json');assert receipt['status']=='PASS' and receipt['manifest_sha256']==sha(ROOT/'preview/manifest.json')
    out=fresh(dest or ROOT/'controls');reg,_=checked_parent();ctx=prepare();records=[];arrays={}
    # Registration precedes controls; no motion-based threshold or candidate search tuning.
    write(out/'registration.json',dict(protocol=CFG,cases=[dict(key=c['key'],source_uid=c['unit']['source_uid'],source_interval=c['unit']['source_interval'],hold=c['hold']['cells_half_open']) for c in reg['cases']],paired_population_sha256=CFG['paired_population_sha256'],scope='AUTOMATIC_SENSITIVITY_NOT_HUMAN_PREFERENCE'))
    for c in reg['cases']:
        s=c['unit']['source_uid'];a,b=c['unit']['source_interval'];source=ctx['lib'].rotations[s];base=source[a:b];model=ReferenceCost(source[max(0,a-30):min(len(source),b+30)],ctx['J'],ctx['parents'])
        for label,R,q,mask,scope in perturbations(source,c):
            assert np.isfinite(R).all() and np.max(abs(R.swapaxes(-1,-2)@R-np.eye(3)))<1e-6
            assert np.array_equal(R[~mask],base[~mask])
            n=c['key']+'__'+label;arrays[n+'__R']=R
            if q is not None:arrays[n+'__q']=q
            legal=None if q is None else bool((np.diff(q)>=1/1.33-1e-8).all() and (np.diff(q)<=1/.75+1e-8).all())
            if label in ['LegalHoldExtension','LegalRateCorner']:assert legal
            if label=='TimeSkip':assert not legal
            d,_=dynamics(R,ctx);hold=None if q is None else float(np.diff(np.interp(c['hold']['cells_half_open'],q,np.arange(len(q))))[0]/30)
            records.append(dict(case_key=c['key'],source_uid=s,label=label,scope=scope,frames=len(R),rotation_sha256=array_hash(R),clock_rate_legal=legal,write_frame_indices=np.flatnonzero(mask).tolist(),changed_frame_count=int(np.any(R!=base,axis=(1,2,3)).sum()),outside_write_mask_exact=True,source_hold_mapped_seconds=hold,GTWLocal=model.score(R,only='GTWLocal')[0]['GTWLocal'],dynamics=d,MotionCritic=None,quality='NOT_EVALUABLE'))
        print('AUTO_PERTURB_SOURCE',s,flush=True)
    comparisons=[]
    for c in reg['cases']:
        own=[r for r in records if r['case_key']==c['key']];base=next(r for r in own if r['label']=='SourceOriginal')
        for r in own:
            comparisons.append(dict(case_key=c['key'],source_uid=r['source_uid'],label=r['label'],FK_D3_delta=r['dynamics']['FK_D3_q95']-base['dynamics']['FK_D3_q95'],angular_acc_delta=r['dynamics']['SO3_transported_acceleration_q95']-base['dynamics']['SO3_transported_acceleration_q95'],high_frequency_delta=None if r['dynamics']['FK_velocity_high_frequency_fraction_above6Hz'] is None or base['dynamics']['FK_velocity_high_frequency_fraction_above6Hz'] is None else r['dynamics']['FK_velocity_high_frequency_fraction_above6Hz']-base['dynamics']['FK_velocity_high_frequency_fraction_above6Hz'],hold_seconds_delta=None if r['source_hold_mapped_seconds'] is None else r['source_hold_mapped_seconds']-base['source_hold_mapped_seconds'],GTWLocal_delta=r['GTWLocal']-base['GTWLocal'],human_preference=None))
    paired_rows,paired_summary=paired();save_arrays(out/'arrays.npz',arrays);write(out/'perturbations.json',records);write(out/'sensitivity.json',comparisons);write(out/'paired_scores.json',paired_rows);write(out/'paired_summary.json',paired_summary)
    write(out/'qualification.json',dict(known_motion_interventions=len(records),source_recordings=len(reg['cases']),real_paired_groups=sum(r['status']=='OBSERVED' for r in paired_rows),independent_new_groups=0,music_artistic_validity='NOT_EVALUABLE',MotionCritic='NOT_EVALUABLE_INPUT_CONTRACT',BAP='NOT_EVALUABLE_NO_REGISTERED_TARGET_ANCHORS',human_records=0,perceptual_winner=None,joint_execution='NOT_STARTED',whole_song='NOT_STARTED',training_steps=0,can_route=False));seal(out);print('AUTO_EVAL_CONTROLS_READY',flush=True)

if __name__=='__main__':run()
