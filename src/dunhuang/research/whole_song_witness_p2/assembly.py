import numpy as np
from .common import array_hash,identity

def assemble(ids,rows,bodies,edges,bridges,frames):
    """Write every actual frame once. UNKNOWN, failed and policy-excluded edges reject."""
    if len(ids)!=13 or len(set(ids))!=13:raise ValueError('Frozen 13-Core slot path required')
    R=np.full((frames,22,3,3),np.nan);coverage=np.zeros(frames,np.int16)
    kind=np.full(frames,-1,np.int16);owner=np.full(frames,-1,np.int16);regions=[];used=[]
    def put(a,b,Q,k,index):
        if not 0<=a<b<=frames or Q.shape!=(b-a,22,3,3) or not np.isfinite(Q).all() or coverage[a:b].any():raise ValueError('Missing, overlapping or invalid actual frame support')
        R[a:b]=Q;coverage[a:b]+=1;kind[a:b]=k;owner[a:b]=index;regions.append(dict(kind='core' if k==0 else 'bridge',index=index,interval=[a,b]))
    for k,u in enumerate(ids):
        row=rows[u];a,b=row['target_interval'];put(a,b,bodies[u],0,k)
        if k+1<len(ids):
            v=ids[k+1];end=rows[v]['target_interval'][0];e=edges.get((u,v))
            if end-b!=48 or e is None or not e.get('measured') or not e.get('eligible') or not e.get('policy_admissible'):raise ValueError('No eligible measured 48-frame edge')
            Q=bridges[e['boundary_key']]
            if array_hash(Q)!=e['numeric']['bridge_sha256']:raise ValueError('Measured bridge identity differs')
            put(b,end,Q,1,k);used.append(dict(left=u,right=v,boundary_key=e['boundary_key'],bridge_sha256=array_hash(Q),interval=[b,end],method='G1'))
    if not np.all(coverage==1) or not np.isfinite(R).all():raise ValueError('Incomplete whole-song coverage')
    for u in ids:
        a,b=rows[u]['target_interval'];assert np.array_equal(R[a:b],bodies[u])
    return R,dict(regions=regions,edges=used,coverage_exact_once=True,path_identity=identity(ids),rotation_sha256=array_hash(R)),kind,owner

def changes(a,b,rows):
    out=[]
    for k,(u,v) in enumerate(zip(a,b)):
        x,y=rows[u],rows[v];same=x['event_uid']==y['event_uid'] and x['source_interval']==y['source_interval']
        out.append(dict(slot=k,left=u,right=v,same_source_unit=same,source_changed=not same,
                        warp_changed_with_same_source=bool(same and x['time_warp']!=y['time_warp']),
                        actual_core_changed=x['rotation_sha256']!=y['rotation_sha256'],
                        left_neighbor_changed=bool(k and a[k-1]!=b[k-1]),right_neighbor_changed=bool(k+1<len(a) and a[k+1]!=b[k+1])))
    return out
