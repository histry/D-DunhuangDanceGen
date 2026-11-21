"""Actual local G1 remeasurement against fixed historical neighbors, no fallback."""
import numpy as np
from dunhuang.research.duration_r62.cross_source import make_observed,solve_boundary,measure
from .contracts import array_hash

def connection(left,right,J,parents,reference):
    obs,gap=make_observed(left,right,48,context=4);curves,proof=solve_boundary(obs,gap,controls=False);Q=curves['G1']
    r=dict(method='G1',gap=48,proof=proof,left_rotation_sha256=array_hash(left),right_rotation_sha256=array_hash(right),
           boundary_readset_sha256=array_hash(np.concatenate((left[-4:],right[:4]))),previous_qualification_inherited=False,
           fallback_used=False,can_route=False,native24='NOT_EVALUABLE')
    if Q is None:return dict(r,constructible=False,passes_original_screen=False,reason='ORIGINAL_G1_CONSTRUCTION_REJECTED')
    full=np.concatenate((left,Q[gap[0]:gap[1]],right));observed=np.concatenate((left,np.full((48,22,3,3),np.nan),right))
    return dict(r,**measure(full,observed,J,parents,(len(left),len(left)+48),reference),bridge_sha256=array_hash(Q[gap[0]:gap[1]]))
