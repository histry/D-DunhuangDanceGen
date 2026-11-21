"""Thin execution adapter: frozen P2 assembly/metrics and frozen P1 G1 only."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .common import *
from dunhuang.research.whole_song_witness_p2.assembly import assemble
from dunhuang.research.whole_song_witness_p2.metrics import evaluate,music_diagnostic
from dunhuang.research.whole_song_witness_p2.study import simple_node
from dunhuang.research.context_feasibility_p1.geometry import solve
from dunhuang.research.duration_r62.cross_source import make_observed,measure
from dunhuang.research.warp_coupling_r53.population import Population

def motion_key(song,ids):return 'motion_'+identity(dict(song=song,ids=ids))

def execute(plans,p,z,rows,edges):
    pop=Population();J,parents=z['J'],z['parents'];assert np.array_equal(J,pop.net.joints) and np.array_equal(parents,pop.net.parents)
    arrays=dict(J=J,parents=parents);bodies={};q={};required={};bridges={};materialized=[]
    for plan in plans:
        ids=plan['ids']
        if ids is None:continue
        for u in ids:
            if u not in bodies:bodies[u],q[u]=reconstruct(rows[u],z,pop.lib.rotations[rows[u]['source_uid']])
        for u,v in zip(ids,ids[1:]):
            edge=edges[u,v];assert edge['measured'] and edge['eligible'] and edge['policy_admissible']
            key=boundary_key(bodies[u],bodies[v],J,parents,pop.reference);assert key==edge['boundary_key']
            if key in required:assert required[key]['numeric']['bridge_sha256']==edge['numeric']['bridge_sha256']
            required[key]=edge
    assert len(required)<=CFG['materialization_budget_per_stage']
    jobs=[(key,bodies[e['left']],bodies[e['right']],J,parents,pop.reference) for key,e in sorted(required.items())]
    with ProcessPoolExecutor(max_workers=CFG['workers']) as pool:
        for i,(key,numeric,R) in enumerate(pool.map(solve,jobs,chunksize=2)):
            old=required[key]['numeric'];assert R is not None and numeric['passes_original_screen']
            assert numeric['bridge_sha256']==old['bridge_sha256'] and numeric['metrics']==old['metrics']
            bridges[key]=R;arrays['bridge_'+key]=R;materialized.append(dict(key=key,bridge_sha256=array_hash(R),numeric_metrics_exact=True))
            if not i%25:print('P3_MATERIALIZE',i+1,len(jobs),flush=True)
    motions={};metrics={};sources={};catalog={s['song']:read(f"outputs/music_events_01/main/{s['song']}/catalog.json") for s in p['songs']}
    for plan in plans:
        ids=plan['ids']
        if ids is None:continue
        s=plan['song'];key=motion_key(s,ids)
        if key in motions:continue
        R,meta,kind,owner=assemble(ids,rows,bodies,edges,bridges,p['songs'][s]['frames']);P,curves,dynamic=evaluate(R,J,parents,kind,owner)
        positions=np.full(len(R),np.nan)
        for u in ids:
            node=rows[u];a,b=node['target_interval'];positions[a:b]=q[u];event=node['event_uid']
            if event not in sources:
                lo,hi=node['source_interval'];sources[event]=simple_node(node);arrays['source_'+identity(event)+'_P']=relative_fk(pop.lib.rotations[node['source_uid']][lo:hi],J,parents).astype('<f4')
        arrays[key+'_R']=R;arrays[key+'_P']=P.astype('<f4');arrays[key+'_kind']=kind;arrays[key+'_owner']=owner;arrays[key+'_source_frame']=positions
        for name,x in curves.items():arrays[key+'_'+name]=x
        seams=[]
        for region in meta['regions']:
            if region['kind']!='bridge':continue
            a,b=region['interval'];observed,gap=make_observed(R[:a],R[b:],48,context=4);part=R[a-4:b+4]
            actual=measure(part,observed,J,parents,gap,pop.reference);assert actual['passes_original_screen']
            seams.append(dict(interval=[a,b],numerical=actual,global_D3_crossing_stencils=[list(range(i,i+4)) for cut in (a,b) for i in range(cut-3,cut)]))
        motions[key]=dict(key=key,song=s,frames=len(R),**meta,nodes=[simple_node(rows[u]) for u in ids],source_cores_exact=True,all_original_G1_bridge_hashes_exact=True)
        metrics[key]=dict(dynamic=dynamic,seams=seams,music=music_diagnostic(P,catalog[s]),metric_scope='ROOT_RELATIVE_BODY22_NOT_NATIVE24_OR_ARTISTIC_QUALITY')
    return arrays,motions,metrics,sources,materialized,pop.binding
