import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.manifold_bridge_r47.geometry import coefficients,jet,jacobian,polynomial_max,segmented,spline
from dunhuang.research.event_coverage_r47.index import merge_protected

def test_body_jet_and_right_jacobian_match_finite_rotations():
    r=np.array([1.,-.7,.5]);w0=np.array([.08,.03,-.04]);w1=np.array([-.03,.1,.02]);a0=np.array([.01,-.02,.005]);a1=-a0;T=8.;c=coefficients(r,w0,a0,w1,a1,T)
    for u,expected in [(0,(w0,a0)),(1,(w1,a1))]:
        actual=jet(c,u,T);np.testing.assert_allclose(actual,expected,atol=1e-10)
    for u in (.1,.5,.9):
        h=1e-5;v=lambda t:np.polynomial.polynomial.polyval(t,c);q=Rotation.from_rotvec(v(u));qnext=Rotation.from_rotvec(v(u+h/T));numerical=(q.inv()*qnext).as_rotvec()/h;w,aa=jet(c,u,T);np.testing.assert_allclose(w,numerical,atol=2e-6);np.testing.assert_allclose(aa,(jet(c,u+h/T,T)[0]-jet(c,u-h/T,T)[0])/(2*h),atol=2e-6)

def test_continuous_maximum_dominates_dense_samples():
    c=coefficients(np.array([.8,0,.2]),np.array([.3,0,0]),np.array([.1,.03,0]),np.array([0,.2,0]),np.array([0,0,.1]),7);actual=np.polynomial.polynomial.polyval(np.linspace(0,1,10001),c).T;assert polynomial_max(c)+1e-10>=np.linalg.norm(actual,axis=-1).max()

def test_segmented_bridge_reads_only_six_observations_and_keeps_visible():
    t=np.arange(256)/80;R=Rotation.from_rotvec(np.c_[np.sin(t)*.5,t*.3,np.cos(t)*.2]).as_matrix();R=np.repeat(R[:,None],22,axis=1);a,ca=segmented(R);hidden=R.copy();hidden[104:152]=np.nan;b,cb=segmented(hidden);assert a is not None and ca==cb;np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a[np.r_[:104,152:256]],R[np.r_[:104,152:256]]);assert ca['physical_boundary_jet_residual']<1e-7;np.testing.assert_allclose(np.linalg.det(a),1,atol=1e-10)

def test_spline_hidden_invariance_and_domain_is_separate_from_jet_qualification():
    R=np.broadcast_to(np.eye(3),(256,22,3,3)).copy();R[104:152]=np.nan;a,c=spline(R);assert a is not None;assert 'NOT_BOUNDARY_JET_QUALIFIED' in c['status'];np.testing.assert_allclose(a,np.broadcast_to(np.eye(3),a.shape),atol=1e-10)

def test_long_hold_is_not_cut_by_preferred_duration():
    assert merge_protected([[120,190],[190,256],[256,310],[310,380],[380,450]],[[152,343]])==[[120,380],[380,450]]

def test_clock_tie_reference_and_unobserved_audio_are_not_hard_negatives():
    from dunhuang.research.music_phase_r47.phase_audit import rank_zero,stage_score,union
    assert rank_zero([0]*5)['recall_at1']==.2
    assert union([[0,100],[50,150],[160,200]])==[[0,150],[160,200]]
    signal=dict(onset=[0]*30,beat_strength=[0]*30,energy=[0]*30)
    assert stage_score(signal,[3],['speed_peak'],-8) is None
