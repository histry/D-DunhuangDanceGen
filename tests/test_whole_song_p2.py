import numpy as np
import pytest
from dunhuang.research.whole_song_witness_p2.assembly import assemble,changes
from dunhuang.research.whole_song_witness_p2.metrics import partition
from dunhuang.research.whole_song_witness_p2.common import array_hash

def fixture():
    ids=['u'+str(k) for k in range(13)];rows={};bodies={};edges={};bridges={}
    I=np.broadcast_to(np.eye(3),(48,22,3,3)).copy()
    for k,u in enumerate(ids):
        a=k*56;rows[u]=dict(target_interval=[a,a+8],source_interval=[0,8],event_uid=u,time_warp=[[0,a],[7,a+7]],rotation_sha256=array_hash(I[:8]))
        bodies[u]=I[:8].copy()
        if k:
            key=str(k);bridges[key]=I.copy();edges[ids[k-1],u]=dict(measured=True,eligible=True,policy_admissible=True,boundary_key=key,numeric=dict(bridge_sha256=array_hash(I)))
    return ids,rows,bodies,edges,bridges,680

def test_exact_complete_frame_assignment_and_core_preservation():
    ids,rows,bodies,edges,bridges,N=fixture();R,m,kind,owner=assemble(ids,rows,bodies,edges,bridges,N)
    assert len(R)==N and m['coverage_exact_once'] and len(m['edges'])==12
    assert np.sum(kind==0)==104 and np.sum(kind==1)==576
    for u in ids:
        a,b=rows[u]['target_interval'];assert np.array_equal(R[a:b],bodies[u])

def test_unknown_edge_and_bridge_hash_change_fail_closed():
    ids,rows,bodies,edges,bridges,N=fixture();missing=dict(edges);missing.pop((ids[4],ids[5]))
    with pytest.raises(ValueError,match='eligible measured'):assemble(ids,rows,bodies,missing,bridges,N)
    bridges['4'][0,0,0,0]=.99
    with pytest.raises(ValueError,match='identity differs'):assemble(ids,rows,bodies,edges,bridges,N)

def test_final_hole_is_not_silently_filled():
    ids,rows,bodies,edges,bridges,N=fixture()
    with pytest.raises(ValueError,match='Incomplete'):assemble(ids,rows,bodies,edges,bridges,N+1)

def test_derivative_stencils_partition_by_actual_full_support():
    kind=np.array([0]*5+[1]*4+[0]*5);owner=np.array([0]*9+[1]*5)
    for n in (1,2,3):
        masks=partition(kind,owner,n);assert np.all(sum(v.astype(int) for v in masks.values())==1)
        cross=np.flatnonzero(masks['cross_region'])
        assert cross.tolist()==[i for i in range(len(kind)-n) if i<5<=i+n or i<9<=i+n]

def test_source_warp_and_neighbor_changes_are_separate():
    ids,rows,*_=fixture();alt=dict(rows);alt['v']=dict(rows['u5'],time_warp=[[0,280],[3,284],[7,287]])
    a=ids.copy();b=ids.copy();b[5]='v';out=changes(a,b,alt)
    assert out[5]['warp_changed_with_same_source'] and not out[5]['source_changed']
    assert out[4]['right_neighbor_changed'] and out[6]['left_neighbor_changed']
