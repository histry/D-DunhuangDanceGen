import copy,itertools,random
import pytest
from dunhuang.research.minimal_context_p3.optimal import solve,verify,node_cost,constrained

def row(group=0,q='q0'):
    return dict(source_uid='recording'+str(group),event_uid='unit'+str(group),source_interval=[0,100],q_sha256=q,rotation_sha256='R'+str(group)+q)

def test_exact_labels_match_exhaustive_paths_on_finite_graphs():
    rng=random.Random(37)
    for trial in range(20):
        layers=[[f'{k}_{j}' for j in range(3)] for k in range(4)];rows={u:row(0 if j<2 else 1,'q'+str(j%2)) for k,layer in enumerate(layers) for j,u in enumerate(layer)}
        baseline=[layer[0] for layer in layers];edges={(u,v) for left,right in zip(layers,layers[1:]) for u in left for v in right if rng.random()<.5}
        for objective in ('source_warp','source_only'):
            cert=solve(layers,edges,rows,baseline,objective);choices=[]
            for path in itertools.product(*layers):
                if all((u,v) in edges for u,v in zip(path,path[1:])):
                    parts=[node_cost(rows[u],rows[b],objective) for u,b in zip(path,baseline)]
                    cost=tuple(sum(x[j] for x in parts) for j in range(len(parts[0])));choices.append((cost,path))
            best=min(choices) if choices else None
            assert cert['ids']==(list(best[1]) if best else None)
            assert cert['cost']==(list(best[0]) if best else None)
            assert verify(cert,layers,edges,rows,baseline)

def test_source_change_priority_cannot_be_outweighed_by_clock_savings():
    layers=[['a','b'],['c','d']];rows=dict(a=row(0,'q1'),b=row(1),c=row(0,'q1'),d=row(),r0=row(),r1=row())
    cert=solve(layers,{('a','c'),('b','d')},rows,['r0','r1'])
    assert cert['ids']==['a','c'] and cert['cost']==[0,2]

def test_warp_secondary_changes_tie_without_changing_source_minimum():
    layers=[['a','b'],['c']];rows=dict(a=row(0,'q1'),b=row(),c=row(),r0=row(),r1=row());edges={('a','c'),('b','c')}
    assert solve(layers,edges,rows,['r0','r1'],'source_only')['ids']==['a','c']
    assert solve(layers,edges,rows,['r0','r1'],'source_warp')['ids']==['b','c']

def test_clock_identity_uses_actual_q_not_redundant_knot_lists():
    a=row();b=dict(a,time_warp=[[0,0],[50,50],[99,99]])
    assert node_cost(a,b,'source_warp')==(0,0)
    b['q_sha256']='different_actual_positions';assert node_cost(a,b,'source_warp')==(0,1)

def test_center_and_absent_edges_are_not_relaxed():
    rows=dict(a=row(),b=row(1),c=row());layers=constrained([['a','b'],['c']],0,['b'])
    cert=solve(layers,set(),rows,['a','c'])
    assert cert['ids'] is None and cert['status']=='NO_COMPLETE_PATH_IN_FROZEN_MEASURED_GRAPH'
    assert solve(layers,{('b','c')},rows,['a','c'])['ids']==['b','c']

def test_tampered_bellman_labels_are_rejected():
    layers=[['a'],['b']];rows=dict(a=row(),b=row());cert=solve(layers,{('a','b')},rows,['a','b'])
    changed=copy.deepcopy(cert);changed['labels'][1]['b']['cost'][0]+=1
    with pytest.raises(ValueError,match='Bellman'):verify(changed,layers,{('a','b')},rows,['a','b'])
    changed=copy.deepcopy(cert);changed['labels'][1]['b']['predecessor']='unknown'
    with pytest.raises(ValueError,match='Bellman'):verify(changed,layers,{('a','b')},rows,['a','b'])
