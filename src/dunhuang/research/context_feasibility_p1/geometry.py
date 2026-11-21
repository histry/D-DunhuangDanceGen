"""Materialize the same frozen G1 on its complete four-frame read/evaluation set."""
import numpy as np
from dunhuang.research.duration_r62.cross_source import make_observed,solve_boundary,measure
from .common import array_hash

def solve(args):
    key,L,R,J,parents,reference=args
    observed,gap=make_observed(L,R,48,context=4);curves,proof=solve_boundary(observed,gap,controls=False);Q=curves['G1']
    result=dict(proof=proof,boundary_readset_sha256=array_hash(np.concatenate((L[-4:],R[:4]))),previous_qualification_inherited=False,
                evaluation_support='4_LEFT_48_GAP_4_RIGHT; ALL_ORIGINAL_GAP_STENCILS_AVAILABLE',method='G1',can_route=False,native24='NOT_EVALUABLE')
    if Q is None:return key,dict(result,constructible=False,passes_original_screen=False,bridge_sha256=None),None
    result.update(measure(Q,observed,J,parents,gap,reference),bridge_sha256=array_hash(Q[4:52]))
    return key,result,Q[4:52].copy()
