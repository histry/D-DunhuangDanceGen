"""Fixed-source 1--3 node common-bank shadow experiment, not formal routing."""
import itertools
import numpy as np
from fractions import Fraction
from .common import CFG,prepare,read,SOURCE_TIME
from .events import motion_catalog,select_nodes,music_catalog
from .alignment import score,shuffled,options
from dunhuang.planning.anchor_reachability import multiple
from dunhuang.research.paper1_auto_eval import music as oldmusic
from dunhuang.research.paper1_auto_eval.experiment import dynamics
from dunhuang.research.gtw_kinematic_baseline.features import ReferenceCost
from dunhuang.research.source_time_q01.study import connect
from dunhuang.research.complete_r36.core import rotation_sample
from dunhuang.research.warp_coupling_r53.common import identity,array_hash

def register(ctx):
    prior=read(SOURCE_TIME/'preview/registration.json');cats=oldmusic.catalogs();catalogs={};cases=[]
    for c in prior['cases']:
        s=c['sources'][0];source=ctx['lib'].rotations[s['source_uid']]
        if s['source_uid'] not in catalogs:catalogs[s['source_uid']]=motion_catalog(source,s['source_uid'])
        nodes=select_nodes(catalogs[s['source_uid']]['events'],s['source_interval']);m=music_catalog(cats[c['music_song']]);T=c['target_frames'];A=c['music_start_frame'];opts=options(nodes,m,s['source_interval'],T,A)
        for k in CFG['node_counts']:
            cases.append(dict(key=c['key']+'__nodes'+str(k),parent_case_key=c['key'],source_uid=s['source_uid'],source_interval=s['source_interval'],target_frames=T,neighbor=c['neighbor'],music_song=c['music_song'],music_start_frame=A,nodes=nodes[:k],requested_nodes=k,status='REGISTERED' if len(nodes)>=k else 'MISSING_NODES_NO_REPLACEMENT',options=opts[:k],music_events=[e for e in m if A/30<e['time']<(A+T-1)/30],music_audio_sha256=c['audio_sha256']))
    return cases,list(catalogs.values())

def run(cases,ctx):
    prior=read(SOURCE_TIME/'preview/candidates.json');oldz=np.load(SOURCE_TIME/'preview/arrays.npz',allow_pickle=False);rows=[];arrays={};cache={};models={}
    for c in cases:
        if c['status']!='REGISTERED':
            a,b=c['source_interval'];arrays[c['key']+'__raw_R']=ctx['lib'].rotations[c['source_uid']][a:b];continue
        a,b=c['source_interval'];T=c['target_frames'];A=c['music_start_frame'];source=ctx['lib'].rotations[c['source_uid']];baseq=np.linspace(a,b-1,T);baseR=rotation_sample(source,baseq);sk=(c['source_uid'],a,b)
        if sk not in models:models[sk]=ReferenceCost(source[max(0,a-30):min(len(source),b+30)],ctx['J'],ctx['parents'])
        bank={};rejected=0
        def add(q,label,targets=None,certificate=None):
            found=next((r for r in bank.values() if np.allclose(q,r['q'],atol=1e-9,rtol=0)),None)
            if found is not None:
                found['labels'].append(label)
                if targets is not None:
                    found['bindings'].append(dict(targets=targets,certificate=certificate))
                    if found['targets'] is None:found['targets']=targets;found['certificate']=certificate
                return
            bank[array_hash(q)]=dict(q=q,labels=[label],targets=targets,certificate=certificate,bindings=[] if targets is None else [dict(targets=targets,certificate=certificate)])
        add(baseq.copy(),'Uniform');ew=next((r for r in prior if r['case_key']==c['parent_case_key'] and r['source_order']==0 and 'E1W' in r['labels']),None)
        if ew is not None:add(oldz[ew['uid']+'__q'].copy(),'E1W')
        combinations=list(itertools.product(*c['options'])) if c['options'] and all(c['options']) else [];assert len(combinations)<=CFG['maximum_combinations_per_task'];guards=[(Fraction(a)+Fraction(i*(b-a-1),T-1),i) for i in [1,2,3,T-4,T-3,T-2]]
        for proposal in combinations:
            if not np.all(np.diff([e['local_frame'] for e in proposal])>0):rejected+=1;continue
            anchors=[dict(source_frame=e['frame'],requested=v['local_frame'],tolerance=CFG['construction_tolerance_frames']) for e,v in zip(c['nodes'],proposal)];cert=multiple([a,b],T,anchors,fixed_knots=guards,additional_constraints=['ACTUAL_SOURCE_SAMPLING','G1'])
            if cert['witness_clock'] is None:rejected+=1;continue
            w=np.asarray(cert['witness_clock']);q=np.interp(np.arange(T),w[:,1],w[:,0]);q[:4]=baseq[:4];q[-4:]=baseq[-4:];add(q,'GuardedRateWitness',[v for v in proposal],cert)
        for h,record in bank.items():
            q=record.pop('q');R=rotation_sample(source,q);uid=identity([c['key'],h]);assert np.all(np.diff(q)>=1/1.33-1e-8) and np.all(np.diff(q)<=1/.75+1e-8);full,connection=connect(ctx,R,c['neighbor'],T,cache);arrays[uid+'__q']=q;arrays[uid+'__R']=R
            if full is not None:arrays[uid+'__full_R']=full
            query=[dict(e,time=float(np.interp(e['frame'],q,np.arange(T)))/30+A/30) for e in c['nodes']];targets=record['targets'];errors=None if targets is None else [abs(e['time']*30-A-v['local_frame']) for e,v in zip(query,targets)];scores={m:score(query,shuffled(c['music_events']) if m=='TypeShuffled' else c['music_events'],typed=m!='NoTypes') for m in ['RoleOrdered','NoTypes','TypeShuffled']};d,_=dynamics(R,ctx)
            rows.append(dict(record,uid=uid,case_key=c['key'],source_uid=c['source_uid'],source_interval=[a,b],clock_sha256=h,rotation_sha256=array_hash(R),full_rotation_sha256=None if full is None else array_hash(full),GTWLocal=models[sk].score(R,only='GTWLocal')[0]['GTWLocal'],clock_RMS=float(np.sqrt(np.mean(((q-baseq)/(b-a-1))**2))),connection=connection,G1_pass=connection['passes_original_screen'],actual_node_target_frames=[e['time']*30-A for e in query],bound_target_error_frames=errors,bound_targets_pass=None if errors is None else bool(max(errors)<=CFG['execution_tolerance_frames']+1e-8),protected_first_last_four_exact=bool(np.array_equal(q[np.r_[0:4,T-4:T]],baseq[np.r_[0:4,T-4:T]]) and np.array_equal(R[np.r_[0:4,T-4:T]],baseR[np.r_[0:4,T-4:T]])),scores=scores,dynamics=d,combinations_registered=len(combinations),combinations_rate_or_order_rejected=rejected,quality='NOT_EVALUABLE'))
        arrays[c['key']+'__raw_R']=source[a:b];print('ROLE_EXECUTION',c['key'],len(bank),'COMBINATIONS',len(combinations),flush=True)
    oldz.close();return rows,arrays,dict(actual_G1_solves=len(cache),candidate_G1_requests=len(rows),shared_common_pool=True)

def choose(cases,rows,methods):
    results=[]
    for c in cases:
        own=[r for r in rows if r['case_key']==c['key']];valid=[r for r in own if r['G1_pass']];key=lambda r,m:(-(r['scores'][m] if r['scores'][m] is not None else -1),r['GTWLocal'],r['clock_RMS'],r['uid'])
        for method in methods:
            selected=None
            if method in ['Uniform','E1W']:selected=next((r for r in own if method in r['labels']),None)
            elif valid:
                if method=='GTWLocal':selected=min(valid,key=lambda r:(r['GTWLocal'],r['clock_RMS'],r['uid']))
                elif method=='SourceAwareSequential':
                    for r in sorted(own,key=lambda r:r['uid']):
                        if r['G1_pass'] and (selected is None or key(r,'RoleOrdered')<key(selected,'RoleOrdered')):selected=r
                else:selected=min(valid,key=lambda r:key(r,method))
            results.append(dict(case_key=c['key'],method=method,selected_uid=None if selected is None else selected['uid'],G1_pass=False if selected is None else selected['G1_pass'],bound_targets_pass=None if selected is None else selected['bound_targets_pass'],nonuniform=None if selected is None else selected['clock_RMS']>1e-9,candidate_budget=len(own),independent_semantic_qualification=False,can_route=False))
    return results
