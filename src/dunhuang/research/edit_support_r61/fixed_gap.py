"""Same-target 48-frame SLERP/G1 comparison, separate from crop validity."""
import inspect
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .common import ROOT, CFG, binding, fresh, read, write, sha, save_arrays, relative_fk
from .intervention import population
from .mask_audit import audit_mask
from dunhuang.research.manifold_bridge_r47.geometry import segmented
from dunhuang.research.bridge_dynamics_r48.study import metrics as original_metrics

FRAMES = 56
GAP = (4, 52)
FIXED = ROOT/'fixed_gap'


def angles(A, B):
    return Rotation.from_matrix((A.swapaxes(-1,-2)@B).reshape(-1,3,3)).magnitude().reshape(A.shape[:-2])


def solve(observed):
    """Read exactly the eight visible poses, ignoring every hidden array value.

    Returns (curves, evidence). G1 rejection remains None; no fallback is used.
    SLERP receives the same observation array but uses its two adjacent endpoints.
    Neither method receives the target phase or hidden truth.
    """
    raw = np.asarray(observed)
    if raw.shape != (FRAMES,22,3,3):
        raise ValueError('Exactly 48 hidden body22 frames and four observations per side required')
    ids = np.r_[0:4,52:56]
    if not np.isfinite(raw[ids]).all():
        raise ValueError('All eight actual boundary observations must be finite')
    visible = np.full((FRAMES,22,3,3),np.nan,dtype=np.float64)
    visible[ids] = raw[ids]
    left,right = visible[3],visible[52]
    delta = Rotation.from_matrix(left.swapaxes(-1,-2)@right).as_rotvec()
    u = np.arange(1,49,dtype=float)/49
    slerp = visible.copy()
    slerp[4:52] = left[None]@Rotation.from_rotvec((u[:,None,None]*delta[None]).reshape(-1,3)).as_matrix().reshape(48,22,3,3)
    G1,guard = segmented(visible,GAP,(4,8,16),np.pi-.05)
    for Q in (slerp,G1):
        if Q is not None:
            assert np.array_equal(Q[ids],visible[ids])
            assert np.isfinite(Q).all()
    return dict(SLERP=slerp,G1=G1),dict(G1=guard,solver_hidden_read=False,
        observed_local_frames=ids.tolist(),edited_local_frames=[4,52],
        SLERP='SHORTEST_ENDPOINT_SO3_INTERPOLATION_ENDPOINTS_3_AND_52',
        G1_rejection_fallback=False,phase_labels_used=False,full_primitive_used=False)


def derivatives(R):
    """Body angular rates and transported successive rates, at 30 Hz."""
    delta=R[:-1].swapaxes(-1,-2)@R[1:]
    omega=Rotation.from_matrix(delta.reshape(-1,3,3)).as_rotvec().reshape(-1,22,3)*30
    accel=((delta[:-1]@omega[1:,...,None])[...,0]-omega[:-1])*30
    return omega,accel


def measure(Q, truth, J, parents, reference, center=28):
    m=original_metrics(Q,J,parents,GAP)
    m.pop('jerk_curve',None)
    P,T=relative_fk(Q,J,parents),relative_fk(truth,J,parents)
    wp,ap=derivatives(Q);wt,at=derivatives(truth)
    alignment=truth.swapaxes(-1,-2)@Q
    dw=(alignment[:-1]@wp[...,None])[...,0]-wt
    da=(alignment[:-2]@ap[...,None])[...,0]-at
    path=np.linalg.norm(np.diff(P[3:53],axis=0),axis=-1).sum(0)
    truth_path=np.linalg.norm(np.diff(T[3:53],axis=0),axis=-1).sum(0)
    activity=np.linalg.norm(wp[3:52],axis=-1).mean(-1)
    truth_activity=np.linalg.norm(wt[3:52],axis=-1).mean(-1)
    passed=m['gap_D2']<=reference['D2'] and m['gap_D3']<=reference['D3'] and m['shoulder_wrist_min_m']>=reference['shoulder_wrist_lower_m']
    return dict(constructible=True,passes_original_screen=bool(passed),
        error_rad=float(angles(truth[4:52],Q[4:52]).mean()),
        fk_error_m=float(np.linalg.norm(P[4:52]-T[4:52],axis=-1).mean()),
        boundary_velocity_truth_error_radps=float(np.linalg.norm(dw[[3,51]],axis=-1).mean()),
        boundary_acceleration_truth_error_radps2=float(np.linalg.norm(da[[2,3,50,51]],axis=-1).mean()),
        target_center_pose_error_rad=float(angles(truth[center],Q[center]).mean()),
        FK_motion_range_rms_m=float(np.sqrt(np.mean(np.var(P[4:52],axis=0)))),
        truth_FK_motion_range_rms_m=float(np.sqrt(np.mean(np.var(T[4:52],axis=0)))),
        FK_path_length_mean_m=float(path.mean()),truth_FK_path_length_mean_m=float(truth_path.mean()),
        maximum_mean_joint_speed_radps=float(activity.max()),truth_maximum_mean_joint_speed_radps=float(truth_activity.max()),
        motion_activity_ratio=float(path.mean()/truth_path.mean()) if truth_path.mean()>1e-9 else None,
        activity_ratio_scope='SOURCE_NORMALIZED_PATH_LENGTH_NOT_ARTISTIC_OR_SEMANTIC_QUALITY',
        target_center_scope='OBSERVED_KINEMATIC_PROXY_CENTER_NOT_VERIFIED_DANCE_KEY_POSE',metrics=m)


def source_balanced(rows, value):
    groups={}
    for row in rows:
        item=row.get(value)
        if item is not None:groups.setdefault(row['source_uid'],[]).append(item)
    v=np.array([np.mean(groups[k]) for k in sorted(groups)],dtype=float)
    if not len(v):return dict(sources=0,tasks=0,mean=None,ci95=None)
    rng=np.random.default_rng(CFG['seed'])
    draws=v[rng.integers(len(v),size=(CFG['bootstrap_draws'],len(v)))].mean(1)
    return dict(sources=len(v),tasks=sum(len(x) for x in groups.values()),mean=float(v.mean()),
        ci95=[float(x) for x in np.quantile(draws,[.025,.975])],
        scope='RECORDING_BALANCED_PAIRED_RECONSTRUCTION; NO_INDEPENDENT_MUSIC_OR_SEMANTIC_GENERALIZATION')


def _phase_band(phase, frames):
    if phase['type']=='speed_peak':return [[max(0,phase['frame']-12),min(frames,phase['frame']+13)]]
    return [phase['evidence_span']]


def run(out, gate=True):
    main_path=ROOT/'main/result.json'
    if gate:
        receipt=read(ROOT/'receipts/main_browser.json')
        if not receipt['passed'] or receipt['main_result_sha256']!=sha(main_path):
            raise ValueError('Actual main source-crop viewer must pass before fixed-gap comparison')
    out=fresh(out);main=read(main_path);assets,parent,fresh_cases=population()
    fresh_index={c['case']:c for c in fresh_cases}
    arrays=dict(J=assets.J,parents=assets.parents);records=[];reference=assets.reference
    for c in main['cases']:
        # Extra visualization/audit metadata in main is not part of source selection.
        original=fresh_index[c['case']]
        assert c['source_uid']==original['source_uid'] and c['selection']==original['selection']
        base=dict(case=c['case'],key='case%d'%c['case'],source_uid=c['source_uid'],dataset=c['dataset'],
            task_label='FIXED_SOURCE_GAP_NOT_A4_VS_A5',source_R_sha256=c['source_R_sha256'],
            human_reviewed=False,verified_phase='UNKNOWN',world_root_contact='UNKNOWN')
        if c['selection']['status']!='MATCHED':
            records.append(dict(base,status='NOT_MATCHABLE_SOURCE_INTERVENTION',selection=c['selection']));continue
        selection=c['selection'];phase=selection['evidence'];center=int(phase['frame']);start=center-28;end=center+28
        component=selection['component']
        if not component[0]<=start<end<=component[1]:
            records.append(dict(base,status='NOT_CONSTRUCTIBLE_FIXED_GAP_SUPPORT',source_interval=[start,end],component=component,target_phase=phase));continue
        uid=c['source_uid'];source=assets.sources[uid];truth=assets.R[uid][start:end].copy()
        assert len(truth)==56
        observed=np.full_like(truth,np.nan);observed[:4]=truth[:4];observed[-4:]=truth[-4:]
        curves,proof=solve(observed)
        key=base['key'];arrays[key+'_truth']=truth;methods={}
        for name,Q in curves.items():
            if Q is None:
                methods[name]=dict(constructible=False,passes_original_screen=False,error_rad=None,fk_error_m=None,
                    arrays_key=None,reason='FROZEN_G1_CHART_OR_JET_REJECTION_NO_FALLBACK')
            else:
                arrays[key+'_'+name]=Q;methods[name]=measure(Q,truth,assets.J,assets.parents,reference)
                methods[name]['arrays_key']=key+'_'+name
        positions=np.arange(start,end,dtype=float);edit=np.zeros(56,bool);edit[4:52]=True
        target_audit=audit_mask(positions,edit,[phase['evidence_span']],_phase_band(phase,len(assets.R[uid])),[center])
        all_audit=audit_mask(positions,edit,[p['evidence_span'] for p in source['phase_evidence']],source['protected_spans'],
                             sorted({int(p['frame']) for p in source['phase_evidence']}))
        paired=all(methods[n]['constructible'] for n in ('SLERP','G1'))
        row=dict(base,status='FIXED_GAP_EVALUATED',source_interval=[start,end],target_interval=[0,56],
            truth_arrays_key=key+'_truth',frames=56,
            source_component=component,edit_source_interval=[center-24,center+24],edit_target_interval=[4,52],
            observed_source_intervals=[[start,start+4],[end-4,end]],source_to_target_clock='IDENTITY_ORIGINAL_30FPS_MINUS_SOURCE_START',
            target_phase=phase,target_phase_audit=target_audit,all_evidence_audit=all_audit,
            methods=methods,solver=proof,mask_scope='RECONSTRUCTION_EDIT_ONLY; NOT_P0_CROP_REMOVAL',
            hidden_truth_identity=dict(source_uid=uid,source_R_sha256=c['source_R_sha256'],interval=[center-24,center+24]),
            G1_minus_SLERP_error_rad=methods['G1']['error_rad']-methods['SLERP']['error_rad'] if paired else None,
            G1_minus_SLERP_fk_error_m=methods['G1']['fk_error_m']-methods['SLERP']['fk_error_m'] if paired else None)
        records.append(row);print('R61_FIXED_GAP',c['case'],uid,methods['G1']['constructible'],flush=True)
    successful=[r for r in records if r['status']=='FIXED_GAP_EVALUATED'];summary={}
    for dataset in ('ALL','ChangE','FineDance','AIST++'):
        rows=successful if dataset=='ALL' else [r for r in successful if r['dataset']==dataset]
        summary[dataset]=dict(tasks=len(rows),paired_constructible=sum(r['G1_minus_SLERP_error_rad'] is not None for r in rows),
            primary=source_balanced(rows,'G1_minus_SLERP_error_rad'),fk=source_balanced(rows,'G1_minus_SLERP_fk_error_m'),
            methods={m:dict(constructed=sum(r['methods'][m]['constructible'] for r in rows),passed_original_screen=sum(r['methods'][m]['passes_original_screen'] for r in rows)) for m in ('SLERP','G1')})
    result=dict(binding=binding(),records=records,summary=summary,main_result_sha256=sha(main_path),
        parent_population_sha256=sha(ROOT.parent/'phase_integrity_r60_02/population/result.json'),
        frozen_geometry_sha256=sha(inspect.getsourcefile(segmented)),reference=reference,
        data_roles='FROZEN_R61_METADATA_ROSTER; NO_NEW_CASE_SELECTION_FROM_RECONSTRUCTION_ERRORS',
        phase_labels_used_by_solvers=False,primary_scope='SAME_48_FRAME_TARGET_METHOD_COMPARISON_NOT_PHASE_INTEGRITY_EFFECT',
        task_masks_reported_as='RECONSTRUCTION_NOT_PHASE_PRESERVATION',human_reviewed=False,
        optimizer_calls=0,training_steps=0,planner_runs=0,can_route=False)
    save_arrays(out/'arrays.npz',arrays);write(out/'result.json',result)
    return result


def verify(out, source=None):
    source=Path(source) if source is not None else FIXED
    expected=read(source/'result.json');current=binding()
    assert current['code']==expected['binding']['code']
    out=fresh(out);actual=run(out/'rebuild',gate=False)
    actual['binding']['commit']=expected['binding']['commit']
    assert actual==expected
    assert sha(source/'arrays.npz')==sha(out/'rebuild/arrays.npz')
    result=dict(result_exact=True,arrays_bytes_exact=True,source_commit=expected['binding']['commit'],
        replay_commit=current['commit'],source_result_sha256=sha(source/'result.json'),optimizer_calls=0,
        fresh_population_and_source_decoding=True,can_route=False)
    write(out/'result.json',result)
    return result
