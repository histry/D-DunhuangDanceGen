"""Frozen comparison rules on one fully executed candidate table."""
from .common import CFG

def basic(r):return bool(r['clock_legal'] and r['moving_support_seconds']>=CFG['minimum_moving_support_seconds'] and r['G1_pass'])
def error(r,target):return abs(r['anchor_target_frame']-target)
def feasible(r,target):return basic(r) and error(r,target)<=CFG['target_error_tolerance_frames']+1e-10
def key(r,target):return (error(r,target),-r['moving_support_seconds'],r['normalized_clock_RMS'],r['source_order'],r['uid'])
def gtw(rows):
    minimum=min(r['GTWLocal'] for r in rows)
    return min([r for r in rows if r['GTWLocal']<=minimum+1e-10],key=lambda r:(r.get('max_clock_shift_frames',r['normalized_clock_RMS']),r['uid']))

def source_order(rows):
    return sorted({r['source_order'] for r in rows},key=lambda k:(-next(r['moving_support_seconds'] for r in rows if r['source_order']==k and 'Uniform' in r['labels']),k))

def choose(method,rows,target):
    """Return actual selection, not a replacement when a constraint fails."""
    order=source_order(rows);consulted=[];selected=None
    if method=='FixedUniform':selected=next(r for r in rows if r['source_order']==0 and 'Uniform' in r['labels'])
    elif method=='FixedE1W':selected=next((r for r in rows if r['source_order']==0 and 'E1W' in r['labels']),None)
    elif method=='FixedGTW':selected=gtw([r for r in rows if r['source_order']==0])
    elif method=='SupportUniform':selected=next(r for r in rows if r['source_order']==order[0] and 'Uniform' in r['labels'])
    elif method=='SequentialGTW':selected=gtw([r for r in rows if r['source_order']==order[0]])
    elif method=='SequentialTargetFallback':
        for source in order:
            own=[r for r in rows if r['source_order']==source];consulted.extend(r['uid'] for r in own);ok=[r for r in own if feasible(r,target)]
            if ok:selected=gtw(ok);break
    elif method=='ExhaustiveSequential':
        # Strong control: enumerate all source groups, retain a global incumbent.
        for source in order:
            for r in sorted([v for v in rows if v['source_order']==source],key=lambda r:r['uid']):
                consulted.append(r['uid'])
                if feasible(r,target) and (selected is None or key(r,target)<key(selected,target)):selected=r
    elif method=='Joint':
        consulted=[r['uid'] for r in rows];ok=[r for r in rows if feasible(r,target)];selected=min(ok,key=lambda r:key(r,target)) if ok else None
    else:raise ValueError(method)
    if not consulted:consulted=[] if selected is None else [selected['uid']]
    # Near miss is explicitly unqualified diagnostic, never counts as a selection.
    diagnostic=min(rows,key=lambda r:(not basic(r),)+key(r,target)) if selected is None else selected
    return dict(method=method,candidate_uid=None if selected is None else selected['uid'],diagnostic_uid=diagnostic['uid'],success=False if selected is None else feasible(selected,target),target_error_frames=None if selected is None else error(selected,target),source_order=None if selected is None else selected['source_order'],consulted_candidate_uids=consulted,full_pool_budget=len(rows),extra_geometry_calls_after_shared_pool=0,selection_score_uses_BAS=False,selection_score_uses_cross_source_GTW=False,failure=None if selected is not None and feasible(selected,target) else 'NO_FEASIBLE_SELECTION_AT_FROZEN_CONSTRAINTS',quality='NOT_EVALUABLE')
