"""Observed-only variable-gap methods; no target pose or phase input."""
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.manifold_bridge_r47.geometry import segmented
from dunhuang.research.bridge_dynamics_r48.study import metrics as original_metrics
from dunhuang.research.edit_support_r61.fixed_gap import angles, derivatives
from .common import relative_fk


def _contract(observed, gap):
    if isinstance(gap, (bool, np.bool_)) or not isinstance(gap, (int, np.integer)) or gap < 1:
        raise ValueError('Positive integer hidden frame count required')
    gap = int(gap)
    raw = np.asarray(observed)
    if raw.shape != (gap+8,22,3,3):
        raise ValueError('Exactly gap+8 body22 frames with four observations per side required')
    ids = np.r_[0:4,gap+4:gap+8]
    if not np.isfinite(raw[ids]).all():
        raise ValueError('All eight actual boundary poses must be finite')
    visible = np.full(raw.shape,np.nan,dtype=np.float64)
    visible[ids] = raw[ids]
    return visible, ids


def solve(observed, gap, controls=False):
    """Same eight available observations, original boundary jets and fixed guards.

    G1 and SingleChartHermite consume exactly the same three nearest poses per
    side to estimate physical body-rate and acceleration. The fourth pose on
    each side is available and preserved, but unused by these frozen estimators.
    SingleChartHermite is the original Lie-quintic with one chart; G1 uses the
    original 4/8/16 reanchored-chart admission schedule. No method falls back.
    """
    visible, ids = _contract(observed,gap)
    interval=(4,4+int(gap));radius=float(np.pi-.05)
    G1,guard=segmented(visible,interval,(4,8,16),radius)
    curves={'G1':G1};guards={'G1':guard}
    if controls:
        left,right=visible[3],visible[gap+4]
        delta=Rotation.from_matrix(left.swapaxes(-1,-2)@right).as_rotvec()
        u=np.arange(1,gap+1,dtype=float)/(gap+1)
        Q=visible.copy()
        Q[4:gap+4]=left[None]@Rotation.from_rotvec((u[:,None,None]*delta[None]).reshape(-1,3)).as_matrix().reshape(gap,22,3,3)
        curves['SLERP']=Q
        guards['SLERP']=dict(status='SHORTEST_ENDPOINT_INTERPOLATION',hidden_read=False,
                             endpoints=[3,gap+4],derivative_constraints='NONE')
        curves['SingleChartHermite'],guards['SingleChartHermite']=segmented(visible,interval,(1,),radius)
    for Q in curves.values():
        if Q is not None:
            assert np.isfinite(Q).all() and np.array_equal(Q[ids],visible[ids])
    return curves,dict(guards=guards,solver_hidden_read=False,phase_labels_used=False,
        observed_local_frames=ids.tolist(),edited_local_frames=list(interval),
        jet_consumed_local_frames=[1,2,3,gap+4,gap+5,gap+6],
        unused_but_preserved_observed_local_frames=[0,gap+7],
        SLERP_consumed_local_frames=[3,gap+4] if controls else [],
        boundary_information_contract='SAME_4_PLUS_4_AVAILABLE; HERMITE_AND_G1_SAME_3_PLUS_3_JETS',
        Hermite_derivative_contract='SAME_OBSERVED_BODY_RATE_ACCELERATION_JETS_AND_RADIUS; SINGLE_CHART_MAY_REJECT',
        rejection_fallback=False,full_primitive_used=False)


def measure(Q, truth, J, parents, reference, gap):
    a,b=4,4+int(gap)
    assert Q.shape==truth.shape==(gap+8,22,3,3)
    m=original_metrics(Q,J,parents,(a,b));m.pop('jerk_curve',None)
    P,T=relative_fk(Q,J,parents),relative_fk(truth,J,parents)
    wp,ap=derivatives(Q);wt,at=derivatives(truth)
    alignment=truth.swapaxes(-1,-2)@Q
    dw=(alignment[:-1]@wp[...,None])[...,0]-wt
    da=(alignment[:-2]@ap[...,None])[...,0]-at
    path=np.linalg.norm(np.diff(P[a-1:b+1],axis=0),axis=-1).sum(0)
    truth_path=np.linalg.norm(np.diff(T[a-1:b+1],axis=0),axis=-1).sum(0)
    activity=np.linalg.norm(wp[a-1:b],axis=-1).mean(-1)
    truth_activity=np.linalg.norm(wt[a-1:b],axis=-1).mean(-1)
    center=a+gap//2
    passed=m['gap_D2']<=reference['D2'] and m['gap_D3']<=reference['D3'] and m['shoulder_wrist_min_m']>=reference['shoulder_wrist_lower_m']
    return dict(constructible=True,passes_original_screen=bool(passed),
        error_rad=float(angles(truth[a:b],Q[a:b]).mean()),
        fk_error_m=float(np.linalg.norm(P[a:b]-T[a:b],axis=-1).mean()),
        boundary_velocity_truth_error_radps=float(np.linalg.norm(dw[[a-1,b-1]],axis=-1).mean()),
        boundary_acceleration_truth_error_radps2=float(np.linalg.norm(da[[a-2,a-1,b-2,b-1]],axis=-1).mean()),
        target_center_pose_error_rad=float(angles(truth[center],Q[center]).mean()),
        FK_motion_range_rms_m=float(np.sqrt(np.mean(np.var(P[a:b],axis=0)))),
        truth_FK_motion_range_rms_m=float(np.sqrt(np.mean(np.var(T[a:b],axis=0)))),
        FK_path_length_mean_m=float(path.mean()),truth_FK_path_length_mean_m=float(truth_path.mean()),
        maximum_mean_joint_speed_radps=float(activity.max()),truth_maximum_mean_joint_speed_radps=float(truth_activity.max()),
        motion_activity_ratio=float(path.mean()/truth_path.mean()) if truth_path.mean()>1e-9 else None,
        activity_ratio_scope='SOURCE_NORMALIZED_PATH_LENGTH_NOT_ARTISTIC_OR_SEMANTIC_QUALITY',
        target_center_scope='SAME_FROZEN_KINEMATIC_PROXY_CENTER; NOT_VERIFIED_DANCE_POSE',metrics=m)


def absent(reason):
    return dict(constructible=False,passes_original_screen=False,error_rad=None,
                fk_error_m=None,arrays_key=None,reason=reason)
