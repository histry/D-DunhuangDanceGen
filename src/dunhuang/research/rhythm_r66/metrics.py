"""R66-B actual-Core diagnostics. Resampling is not learned reperformance.

All outputs are read-only motion/time-map measurements. The caller owns the
analytic PCHIP certificate and objective. No human quality, Bridge, Root/contact
or Native qualification can be inferred from this module.
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation

FRAMES = 240
FPS = 30
_EPS = 1e-9


def _real(value, name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not math.isfinite(value):
        raise ValueError('Finite numeric '+name+' required')
    return float(value)


def _rotations(value, name, frames=None):
    a=np.asarray(value,dtype=float)
    if a.ndim!=4 or a.shape[1:]!=(22,3,3) or len(a)<4 or frames is not None and len(a)!=frames or not np.isfinite(a).all():
        raise ValueError(name+' must be finite actual body22 SO3 observations')
    if np.max(np.abs(a.swapaxes(-1,-2)@a-np.eye(3)))>2e-4 or np.max(np.abs(np.linalg.det(a)-1))>2e-4:
        raise ValueError(name+' does not contain valid SO3 rotations')
    return a


def _interval(value,n):
    if (not isinstance(value,(list,tuple,np.ndarray)) or len(value)!=2
            or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in value)):
        raise ValueError('Integer half-open source interval required')
    a,b=map(int,value)
    if not 0<=a<b<=n or b-a<4:
        raise ValueError('Full genuine source support required')
    return a,b


def _positions(value,a,b):
    q=np.asarray(value,dtype=float)
    if (q.shape!=(FRAMES,) or not np.isfinite(q).all() or not (np.diff(q)>0).all()
            or q[0]<a or q[-1]>b-1 or not np.isclose(q[0],a,atol=_EPS,rtol=0)
            or not np.isclose(q[-1],b-1,atol=_EPS,rtol=0)):
        raise ValueError('240 strictly increasing absolute-source positions and fixed endpoints required')
    if np.floor(q).min()<a or np.ceil(q).max()>=b:
        raise ValueError('Positive-weight read support leaves the full source Core')
    return q


def rotation_sample(refR,positions):
    """The frozen source-SLERP convention, with no invented endpoint poses."""
    q=np.asarray(positions,float);lo=np.floor(q).astype(int);hi=np.minimum(lo+1,len(refR)-1)
    u=(q-lo)[:,None,None]
    rel=refR[lo].swapaxes(-1,-2)@refR[hi]
    v=Rotation.from_matrix(rel.reshape(-1,3,3)).as_rotvec().reshape(len(q),22,3)
    return refR[lo]@Rotation.from_rotvec((v*u).reshape(-1,3)).as_matrix().reshape(len(q),22,3,3)


def _angles(a,b):
    return Rotation.from_matrix((a.swapaxes(-1,-2)@b).reshape(-1,3,3)).magnitude().reshape(a.shape[:-2])


def _fk(R,J,parents):
    p=[np.zeros((len(R),3))];world=[R[:,0]]
    for j in range(1,22):
        k=int(parents[j]);p.append(p[k]+(world[k]@(J[j]-J[k])[None,:,None])[...,0]);world.append(world[k]@R[:,j])
    return np.stack(p,axis=1)


def _summary(x):
    x=np.asarray(x,float).reshape(-1)
    if not len(x):return dict(count=0,mean=None,p50=None,p95=None,maximum=None,minimum=None)
    return dict(count=len(x),mean=float(x.mean()),p50=float(np.quantile(x,.5)),
                p95=float(np.quantile(x,.95)),maximum=float(x.max()),minimum=float(x.min()))


def _motion(R,P,curves=False):
    delta=R[:-1].swapaxes(-1,-2)@R[1:]
    omega=Rotation.from_matrix(delta.reshape(-1,3,3)).as_rotvec().reshape(-1,22,3)*FPS
    alpha=((delta[:-1]@omega[1:,...,None])[...,0]-omega[:-1])*FPS
    jerk=((delta[:-2]@alpha[1:,...,None])[...,0]-alpha[:-1])*FPS
    out=dict(frames=len(R),SO3={},FK={},units=dict(SO3_D1='rad/s',SO3_D2='rad/s^2',SO3_D3='rad/s^3',
        FK_D1='m/s',FK_D2='m/s^2',FK_D3='m/s^3'),
        derivative_scope='SO3_BODY_RATE_COVARIANT_SUCCESSIVE_DIFFERENCES; FK_ACTUAL_FINITE_DIFFERENCES',
        region_scope='WHOLE_SINGLE_CORE; NO_REAL_OUTSIDE_CONTEXT_OR_BRIDGE_CERTIFICATE')
    for order,values in enumerate((omega,alpha,jerk),1):
        x=np.sqrt(np.mean(values**2,axis=(1,2)))
        out['SO3']['D'+str(order)]=_summary(x)
        if curves:out['SO3']['D'+str(order)]['curve']=x.tolist()
        y=np.sqrt(np.mean((np.diff(P,n=order,axis=0)*FPS**order)**2,axis=(1,2)))
        out['FK']['D'+str(order)]=_summary(y)
        if curves:out['FK']['D'+str(order)]['curve']=y.tolist()
    out['endpoint_mean_joint_speed_radps']=dict(left=float(np.linalg.norm(omega[:3],axis=-1).mean()),
                                               right=float(np.linalg.norm(omega[-3:],axis=-1).mean()))
    return out


def _geometry(R,P):
    excursion=_angles(np.broadcast_to(R[0],R.shape),R)
    path=np.linalg.norm(np.diff(P,axis=0),axis=-1).sum(0)
    extent=P.max(axis=0)-P.min(axis=0)
    radial=np.linalg.norm(P-P[0],axis=-1).max(0)
    return dict(SO3_first_pose_excursion_max_rad_per_joint=excursion.max(0).tolist(),
        SO3_first_pose_excursion_max_mean_rad=float(excursion.max(0).mean()),
        FK_axis_extent_m=extent.tolist(),FK_axis_extent_rms_m=float(np.sqrt(np.mean(extent**2))),
        FK_radial_excursion_max_m_per_joint=radial.tolist(),FK_radial_excursion_mean_m=float(radial.mean()),
        FK_path_length_m_per_joint=path.tolist(),FK_path_length_mean_m=float(path.mean()),
        FK_time_weighted_range_rms_m=float(np.sqrt(np.mean(np.var(P,axis=0)))),
        scope='GEOMETRIC_RANGE_AND_SAMPLED_PATH; TIME_WEIGHTED_VARIANCE_CHANGES_WITH_HOLD_DURATION')


def _ratio(a,b):
    return float(a/b) if abs(b)>1e-9 else None


def _retention(actual,reference):
    fields=('SO3_first_pose_excursion_max_mean_rad','FK_axis_extent_rms_m',
            'FK_radial_excursion_mean_m','FK_path_length_mean_m','FK_time_weighted_range_rms_m')
    return dict(ratios={k:_ratio(actual[k],reference[k]) for k in fields},
        scope='SAMPLING_AND_TIME_REWEIGHTING_DIAGNOSTICS_NOT_LEARNED_SOURCE_FOLLOWING_OR_CULTURAL_STYLE',
        near_static_denominator='UNKNOWN_RATIO_NOT_ZERO_OR_ONE',quality_pass=None)


def event_key(event):
    return str(event.get('body_region','UNKNOWN'))+':'+str(event['type'])+':'+str(int(event['frame']))


def _events(events,a,b,n):
    rows=[]
    for p in ([] if events is None else events):
        if not isinstance(p,dict) or not isinstance(p.get('type'),str):raise ValueError('Raw R47 phase records required')
        t=p.get('frame');span=p.get('evidence_span')
        if isinstance(t,(bool,np.bool_)) or not isinstance(t,(int,np.integer)):
            raise ValueError('Actual integer source event frame required')
        if (not isinstance(span,(list,tuple)) or len(span)!=2
                or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) for x in span)
                or not 0<=span[0]<=t<span[1]<=n):raise ValueError('Raw half-open phase evidence span required')
        if a<=t<b:rows.append(dict(p))
    return rows


def _audio_bank(audio_events):
    out={}
    for kind in ('beat','accent','alternate_onset'):
        value=None if audio_events is None else audio_events.get(kind)
        if value is None:out[kind]=None;continue
        x=np.asarray(value,float)
        if x.ndim!=1 or not np.isfinite(x).all() or len(x) and (x[0]<0 or not (np.diff(x)>0).all()):
            raise ValueError('Audio events must be unique increasing absolute frame centers')
        out[kind]=x
    out['accent_weights']=None
    if audio_events is not None and audio_events.get('accent_strength') is not None:
        strength=np.asarray(audio_events['accent_strength'],float)
        if strength.ndim!=1 or not np.isfinite(strength).all() or (strength<0).any():
            raise ValueError('Finite nonnegative absolute-frame accent strength curve required')
        if out['accent'] is not None and len(out['accent']):
            if out['accent'][-1]>len(strength)-1:raise ValueError('Accent strength curve does not cover its raw clock')
            out['accent_weights']=np.interp(out['accent'],np.arange(len(strength)),strength)
    return out


def _alignment(frames,audio):
    frames=np.asarray(frames,float);out={}
    for kind in ('beat','accent','alternate_onset'):
        evidence=audio[kind]
        if not len(frames) or evidence is None or not len(evidence):
            out[kind]=dict(status='NOT_EVALUABLE',query_count=len(frames),evidence_count=0 if evidence is None else len(evidence),
                mean_nearest_error_frames=None,mean_nearest_error_seconds=None,gaussian3_score=None)
            continue
        d=np.abs(frames[:,None]-evidence[None,:]);nearest=d.min(1)
        values=np.exp(-d**2/(2*3.**2))
        if kind=='accent' and audio['accent_weights'] is not None:values*=audio['accent_weights'][None]
        out[kind]=dict(status='PROXY_ONLY',query_count=len(frames),evidence_count=len(evidence),
            mean_nearest_error_frames=float(nearest.mean()),mean_nearest_error_seconds=float(nearest.mean()/FPS),
            gaussian3_score=float(values.max(1).mean()))
    beat,accent=out['beat']['gaussian3_score'],out['accent']['gaussian3_score']
    out['primary_half_beat_half_accent']=(beat+accent)/2 if beat is not None and accent is not None else None
    out['scope']='MANY_TO_NEAREST_AUDIO_PROXY_DIAGNOSTIC_NOT_EVENT_TRUTH_OR_ARTISTIC_QUALITY'
    return out


def measure(R,refR,J,parents,positions,source_interval,phase_config=None,source_events=None,
            audio_events=None,audio_origin_frame=0,optimized_event_ids=(),include_curves=False):
    """Measure one real240-frame output; compare the same source at uniform time.

    `refR` is the full original recording; positions are absolute source frames.
    Audio event lists are absolute raw-audio frames. accent_strength is the full
    raw-audio-frame curve. All four conditions must be assessed with the same
    TRUE audio bank, even when a condition optimized a shifted bank. An optional
    alternate_onset bank is accepted, but this function extracts no new detector.
    `optimized_event_ids` contains event_key strings or the actual raw phase dicts.
    """
    R=_rotations(R,'actual R',FRAMES);refR=_rotations(refR,'full source refR')
    a,b=_interval(source_interval,len(refR));q=_positions(positions,a,b)
    origin=_real(audio_origin_frame,'audio origin frame')
    if origin<0:raise ValueError('Actual audio origin may not be negative')
    J=np.asarray(J,float);parents=np.asarray(parents)
    if J.shape!=(22,3) or not np.isfinite(J).all() or parents.shape!=(22,) or parents.dtype.kind not in 'iu' or parents[0]!=-1 or any(not 0<=parents[j]<j for j in range(1,22)):
        raise ValueError('Actual common22 skeleton and topologically ordered parents required')
    uniform_positions=np.linspace(a,b-1,FRAMES)
    uniform=rotation_sample(refR,uniform_positions)
    expected=rotation_sample(refR,q)
    native=refR[a:b]
    P,U,N=_fk(R,J,parents),_fk(uniform,J,parents),_fk(native,J,parents)
    actual_geometry,uniform_geometry,native_geometry=(_geometry(x,y) for x,y in ((R,P),(uniform,U),(native,N)))
    dq=np.diff(q);d2=np.diff(q,n=2);mid=(dq[:-1]+dq[1:])/2;rho=(FRAMES-1)/(b-a-1)
    tau_rate=1/dq;tau_curvature=-d2/(mid**3)
    reconstruction=_angles(expected,R);endpoints=_angles(np.stack((refR[a],refR[b-1])),R[[0,-1]])
    out=dict(frames=FRAMES,fps=FPS,source_interval=[a,b],target_interval=[0,FRAMES],
        native_source_frames=b-a,source_to_output='MONOTONE_SOURCE_ROTATION_RESAMPLING_NO_NEW_GENERATOR',
        actual=_motion(R,P,include_curves),reference_uniform=_motion(uniform,U,include_curves),
        reference_native=_motion(native,N,include_curves),
        reference_scope='UNIFORM240_FOR_SAME_BUDGET_COMPARISON; NATIVE_SOURCE_DURATION_REPORTED_SEPARATELY',
        geometric_range=dict(actual=actual_geometry,uniform=uniform_geometry,native=native_geometry),
        versus_uniform=_retention(actual_geometry,uniform_geometry),versus_native=_retention(actual_geometry,native_geometry),
        reconstruction_check=dict(mean_angle_rad=float(reconstruction.mean()),maximum_angle_rad=float(reconstruction.max()),
            endpoint_maximum_angle_rad=float(endpoints.max()),
            scope='SAMPLER_CONFORMANCE_ONLY; ZERO_ERROR_BY_CONSTRUCTION_IS_NOT_REPERFORMANCE_OR_STYLE_EVIDENCE'),
        time_map=dict(source_per_target_rate=_summary(dq),target_per_source_rate=_summary(tau_rate),
            uniform_target_per_source_rate=rho,normalized_tau_rate=_summary(tau_rate/rho),
            source_time_curvature_per_second=_summary(d2*FPS),
            inverse_tau_second_derivative_per_source_frame=_summary(tau_curvature),
            normalized_rate_penalty_dense=float(np.mean((tau_rate/rho-1)**2)),
            normalized_curvature_penalty_dense=float(np.mean((FPS*tau_curvature/rho)**2)),
            left_source_per_target_rate=dq[:3].tolist(),right_source_per_target_rate=dq[-3:].tolist(),
            scope='FINITE_DENSE_ESTIMATES_NOT_ANALYTIC_PCHIP_EXTREMA_CERTIFICATE',analytic_rate_certificate=False),
        bridge_recomputed=False,bridge_qualified=None,world_root_contact='UNKNOWN',Native24='NOT_EVALUABLE',
        human_quality_pass=None,artistic_quality='NOT_EVALUABLE',training_steps=0,new_model_samples=0,can_route=False)
    audio=_audio_bank(audio_events)
    phases=_events(source_events,a,b,len(refR))
    if any(not isinstance(v,(dict,str)) for v in optimized_event_ids):
        raise ValueError('Optimizer event identities must be raw phase dicts or event_key strings')
    used={event_key(v) if isinstance(v,dict) else v for v in optimized_event_ids}
    phase_rows=[];mapped_peaks=[]
    for event in phases:
        frame=int(event['frame']);lo,hi=map(int,event['evidence_span'])
        t=float(np.interp(frame,q,np.arange(FRAMES)))
        floor,ceil=np.floor(q),np.ceil(q)
        touched=((floor>=lo)&(floor<hi))|((ceil>=lo)&(ceil<hi))
        fully=((floor>=lo)&(floor<hi))&((ceil>=lo)&(ceil<hi))
        key=event_key(event)
        phase_rows.append(dict(event_key=key,type=event['type'],body_region=event.get('body_region','UNKNOWN'),
            source_frame=frame,source_evidence_span=[lo,hi],target_frame=t,audio_frame=origin+t,
            source_evidence_whole_inside_core=lo>=a and hi<=b,
            target_read_support_touch_frames=int(touched.sum()),target_read_support_fully_inside_frames=int(fully.sum()),
            source_evidence_seconds=(hi-lo)/FPS,target_support_touch_seconds=float(touched.sum()/FPS),
            used_by_warp=key in used,unused_scope='NOT_IN_CALLER_DECLARED_OPTIMIZER_LIST; NOT_PROOF_OF_STATISTICAL_INDEPENDENCE',
            truth='KINEMATIC_PROXY_NOT_SEMANTIC_PHASE_TRUTH'))
        if event['type'].endswith('speed_peak') and lo>=a and hi<=b:mapped_peaks.append(origin+t)
    source_uniform=[origin+(p['frame']-a)*(FRAMES-1)/(b-a-1) for p in phases
                    if p['type'].endswith('speed_peak') and a<=p['evidence_span'][0] and p['evidence_span'][1]<=b]
    out['source_phase_diagnostics']=phase_rows
    out['mapped_source_peaks']=dict(audio_query_frames=mapped_peaks,alignment=_alignment(mapped_peaks,audio),
        uniform_alignment=_alignment(source_uniform,audio),
        scope='OBJECTIVE_RELATED_TECHNICAL_REACHABILITY; IMPROVEMENT_NOT_INDEPENDENT_QUALITY')
    out['hold_turn_diagnostics']=[p for p in phase_rows if p['type'] in ('pose_hold','root_turn_span','direction_turn')]
    if phase_config is None:
        out['actual_output_peaks']=dict(status='NOT_EVALUABLE_MISSING_REGISTERED_PHASE_CONFIG',peaks=[])
    else:
        from dunhuang.research.paired_event_r41.events import profiles,phases as detect
        actual_events=detect(profiles(R),phase_config)
        uniform_events=detect(profiles(uniform),phase_config)
        def peaks(events):
            return [dict(p) for p in events if p['type']=='speed_peak'
                    and p['evidence_span'][0]>=3 and p['evidence_span'][1]<=FRAMES-3]
        actual_peaks,uniform_peaks=peaks(actual_events),peaks(uniform_events)
        out['actual_output_peaks']=dict(status='PROXY_DIAGNOSTIC_ONLY',peaks=actual_peaks,
            alignment=_alignment([origin+p['frame'] for p in actual_peaks],audio),
            uniform_peaks=uniform_peaks,uniform_alignment=_alignment([origin+p['frame'] for p in uniform_peaks],audio),
            boundary_excluded_frames_each_side=3,
            detector='FROZEN_R47_PROFILES_AND_PHASES_ON_ACTUAL240_CORE',additional_smoothing=False,
            source_peak_queries_reused=False,statistical_independence_or_truth=False,
            scope='REDETECTED_OUTPUT_VELOCITY_PEAKS; CORE_ADAPTIVE_PROMINENCE; NOT_OPTIMIZED_EVENT_LOCATIONS_OR_TRUE_ACCENTS')
    return out
