"""UNKNOWN is absent, never implicitly eligible. Shared finite measured graph."""
import numpy as np
from .common import CFG,admissible

def suffix(layers,edges):
    supported=[set() for _ in layers]
    supported[-1]={r['uid'] for r in layers[-1]}
    for k in range(len(layers)-2,-1,-1):
        supported[k]={r['uid'] for r in layers[k] if any((r['uid'],v) in edges for v in supported[k+1])}
    return supported

def sequential(layers,eligible):
    support=suffix(layers,eligible);path=[];extensions=0
    for k,layer in enumerate(layers):
        found=None
        for r in sorted(layer,key=lambda r:(-r['heuristic'],r['uid'])):
            extensions+=1
            if extensions>CFG['path_budget_extensions']:return None,dict(status='BUDGET_EXHAUSTED',extensions=extensions)
            if r['uid'] in support[k] and (not k or (path[-1],r['uid']) in eligible):found=r['uid'];break
        if found is None:return None,dict(status='NO_PATH_IN_MEASURED_GRAPH',extensions=extensions)
        path.append(found)
    return path,dict(status='COMPLETE_MEASURED_PATH',extensions=extensions,exact_suffix_lookahead=True)

def inspect(layers,edges,reference):
    from dunhuang.research.execution_planning_r58.graph import choose
    good={(e['left'],e['right']):e for e in edges if e['eligible']}
    strong,work=sequential(layers,good)
    adapted=[]
    for layer in layers:
        adapted.append([dict(id=r['uid'],event_uid=r['event_uid'],source_uid=r['source_uid'],target=r['target_interval'],heuristic=r['heuristic'],execution_cost=0.) for r in layer])
    graph=dict(layers=adapted,edges=[dict(left=e['left'],right=e['right'],eligible=True,evidence=dict(selected='G1',methods=dict(G1=e['numeric']['metrics']))) for e in good.values()],base_intervals=[layer[0]['target_interval'] for layer in layers])
    if any(not layer for layer in adapted):raise ValueError('Cannot invent a missing node layer')
    choice,stats=choose(graph,np.zeros(layers[-1][0]['target_interval'][1]),reference,fixed=True,execution=False,structure=False)
    expansions=sum(x['expanded'] for x in stats['layers']);assert expansions<=CFG['path_budget_extensions']
    assert bool(suffix(layers,good)[0])==bool(strong)
    return dict(strong_sequential_path=strong,strong_work=work,R58_adapter_path=None if choice is None else choice['ids'],R58_search=stats,
                R58_adapter_scope='ORIGINAL_CHOOSE_WITH_EXECUTION_AND_STRUCTURE_DISABLED; FIXED_P0_LATTICE_NOT_FULL_R58_REPRODUCTION',
                exact_suffix_has_path=bool(suffix(layers,good)[0]),unknown_edges_not_admitted=True,quality='NOT_EVALUABLE')
