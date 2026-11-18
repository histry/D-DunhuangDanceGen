"""Proxy metrics do not replace independent phrase labels or human preference."""
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from dunhuang.research.validation_r51.common import regions
from .slots import centers
from .music import descriptors
from .common import *

def correspondence(pred,ref,tolerance):
 pred=np.asarray(pred);ref=np.asarray(ref)
 if not len(pred) or not len(ref):return dict(predictions=len(pred),references=len(ref),matches=0,precision=0. if len(pred) else None,recall=0. if len(ref) else None,F1=0. if len(pred)+len(ref) else None,matched_MAE_frames=None)
 d=np.abs(pred[:,None]-ref[None]);cost=np.where(d<=tolerance,d,100000);a,b=linear_sum_assignment(cost);good=d[a,b]<=tolerance;n=int(good.sum());return dict(predictions=len(pred),references=len(ref),matches=n,precision=n/len(pred),recall=n/len(ref),F1=2*n/(len(pred)+len(ref)),matched_MAE_frames=float(d[a[good],b[good]].mean()) if n else None)

def evaluate(pop,arrays,plan,spans,music,motion_fk_similarity):
 R,P=arrays['R'],arrays['P'];speed=np.linalg.norm(np.diff(P.astype(float),axis=0,prepend=P[:1])*30,axis=-1).mean(1);peaks=find_peaks(speed,distance=CFG['accent_distance_frames'],prominence=max(float(np.ptp(speed))*.1,1e-8))[0];idx=[e['bank'] for e in plan['events']];ms=kernel(descriptors(music['spec'],spans));ds=motion_fk_similarity[np.ix_(idx,idx)];triangle=np.triu_indices(len(idx),1);a,b=ms[triangle],ds[triangle];corr=float(spearmanr(a,b).statistic) if np.ptp(a)>1e-12 and np.ptp(b)>1e-12 else None;repeat=np.array([plan['events'][i]['event_uid']==plan['events'][j]['event_uid'] for i,j in zip(*triangle)]);error=0.
 for e in plan['events']:
  lo,hi=e['target_interval']
  w=np.array(e['time_warp']['time_warp']);q=np.interp(np.arange(lo,hi),w[:,1],w[:,0]);Q=rotation_sample(pop.lib.rotations[e['source_uid']],q);error=max(error,float(np.max(abs(Q-R[lo:hi]))))
 return dict(phrase_truth='NOT_EVALUABLE',phrase_boundary_proxy=correspondence(centers(spans),music['proxy_boundaries'],CFG['proxy_boundary_tolerance_frames']),motion_accent_proxy={str(t):correspondence(peaks,music['proxy_accents'],t) for t in CFG['accent_tolerances_frames']},repetition_structure_proxy=dict(spearman=corr,repeated_event_pairs=int(repeat.sum()),similarity_for_repeated=float(a[repeat].mean()) if repeat.any() else None,similarity_for_different=float(a[~repeat].mean()) if (~repeat).any() else None,independent_artistic_truth=False),kinematics=regions(R,pop.net.joints,pop.net.parents,spans),source_core_max_matrix_error=error,source_fidelity='CONSTRUCTED_RETENTION_NOT_LEARNED_ACCURACY',human_preference='NOT_EVALUATED',physical_qualified=False)
