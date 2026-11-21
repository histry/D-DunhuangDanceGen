"""Variable-frame adapter of the frozen R53 lattice; no score-based proposals."""
import numpy as np
from dunhuang.research.complete_r36.core import multi_warp
from dunhuang.research.warp_coupling_r53 import warp_lattice as lattice
from dunhuang.planning.anchor_reachability import multiple
from .contracts import CFG,source_reads,identity,array_hash

def build(unit,target,beats):
    source=unit['source_interval'];phases=[dict(p,type=p['body_region']+'_'+p['type']) for p in unit['phases']]
    first=multi_warp(source,target,phases,beats)
    if first is None:return [],dict(reason='ORIGINAL_E1W_INFEASIBLE',legal_path_count=0,scanned=0)
    graph=lattice.lattice(source,target,phases,beats);paths=lattice.best(graph,CFG['scan_paths'])
    if not paths or paths[0]!=first:raise ValueError('R53 baseline differs from original E1W')
    q0=np.interp(np.arange(*target),np.array(first['time_warp'])[:,1],np.array(first['time_warp'])[:,0]);rows=[]
    for rank,value in enumerate(paths):
        w=lattice.validate(value['time_warp'],source,target);q=source_reads(np.interp(np.arange(*target),w[:,1],w[:,0]),source,target)
        if rows and np.allclose(q,q0,atol=1e-9,rtol=0):continue
        cert=multiple(source,target[1]-target[0],[],fixed_knots=[(float(x),float(y-target[0])) for x,y in w])
        if cert['status']!='REACHABLE_BY_RATE':raise ValueError('Executed clock lacks joint rate witness')
        key=dict(source_uid=unit['source_uid'],event_uid=unit['unit_uid'],source_interval=source,target_interval=target,q_sha256=array_hash(q))
        rows.append(dict(**key,uid=identity(key),time_warp=w.tolist(),q=q.tolist(),origin='E1W' if not rows else 'R53_FIRST_DISTINCT',original_cost_rank=rank,certificate=cert,
                         rate_min=float(np.min(np.diff(w[:,1])/np.diff(w[:,0]))),rate_max=float(np.max(np.diff(w[:,1])/np.diff(w[:,0])))))
        if len(rows)==CFG['max_warps']:break
    return rows,dict(legal_path_count=lattice.suffix(graph)[0][0],scanned=len(paths),scope='FIRST64_ORIGINAL_COST_PATHS_NOT_CONTINUOUS_COMPLETE',scores_consulted=False,artificial_fill=0)
