import copy,json,os,sys,subprocess
from pathlib import Path
import numpy as np
import pytest
from dunhuang.io import sha
from dunhuang.planning.planner import plan,propose_candidates
from dunhuang.planning.event_plan import semantic_plan_id,revised_plan
from dunhuang.planning.feedback import jerk_localization
from dunhuang.planning.replanner import budget,variants,execution_key,route,verify_cache,launch,save

def fixture(tmp_path):
    path=tmp_path/'motion.npy';np.save(path,np.ones((300,151),np.float32))
    entries=[dict(event_uid=str(i),source_uid=str(i),source_interval=[0,90],center=45,
        duration_frames=90,reference=dict(path=str(path),sha256=sha(path)),activity_percentile=i/5,
        entry=np.tile(np.eye(3),(8,1,1)).tolist(),exit=np.tile(np.eye(3),(8,1,1)).tolist(),
        feasibility=dict(temporal8='PASS',native='UNKNOWN')) for i in range(5)]
    timing=dict(frames=270,slots=[dict(start=a,end=a+90,density=.5) for a in (0,90,180)],beats=[])
    return dict(entries=entries),timing

def test_single_search_parity_local_revision_and_identity(tmp_path):
    catalog,timing=fixture(tmp_path);parent=plan(catalog,timing,'audio')
    pool=propose_candidates(catalog,timing,'audio')
    assert pool[0]==parent and len(pool)==3
    assert len({semantic_plan_id(p) for p in pool})==3
    revised=propose_candidates(catalog,timing,'audio',fixed_events={i:e for i,e in enumerate(parent['events']) if i!=1},
        excluded_events={1:[parent['events'][1]['event_uid']]})[0]
    assert revised['events'][0]==parent['events'][0]
    assert revised['events'][2]==parent['events'][2]
    assert revised['events'][1]['event_uid']!=parent['events'][1]['event_uid']
    lineage=revised_plan(revised,parent,dict(sha256='feedback'),'REPLAN',[1])
    assert semantic_plan_id(lineage)==semantic_plan_id(revised)!=semantic_plan_id(parent)
    assert lineage['lineage']['event_globally_infeasible'] is False
    changed=copy.deepcopy(parent);changed['events'][0]['expression']['amplitude']=.9
    assert semantic_plan_id(changed)!=semantic_plan_id(parent)

def test_jerk_stencil_owners_include_both_overlap_edges_and_no_padding():
    x=np.zeros((225,24,3),np.float32);x[75:,0,0]=1;x[150:,1,0]=2
    p=dict(events=[dict(music_interval=[0,110]),dict(music_interval=[110,225])])
    out=jerk_localization(x,[x[:150],x[75:225]],p,10)
    assert out['non_boundary_count']==0
    assert out['supporting_raw_window_exceeded_count']==out['violating_stencils']
    assert all(len(r['stencil'])==2 and r['stencil'][1]-r['stencil'][0]==4 for r in out['rows'])
    assert out['causal_attribution']=='UNKNOWN'
    # A last partial window has only real requested frames, not padded owners.
    y=np.zeros((160,24,3));y[159,0,0]=1
    short=jerk_localization(y,[y[:150],y[75:]],p,10)
    assert short['rows'][-1]['window_owners']==[1]

def test_budget_and_default_routing_are_bounded():
    b=budget({})
    fail=dict(native_result='FAIL',broad_failure=True,recommended_route='MODEL_DIAGNOSIS')
    assert route(fail,b,0)=='STOP_MODEL_DIAGNOSIS'
    diagnostic=budget(dict(diagnostic_continue_on_broad_failure=True,max_diagnostic_replans=1))
    assert route(fail,diagnostic,0)=='DIAGNOSTIC_INNER_AND_ONE_REPLAN'
    assert route(fail,diagnostic,1)=='DIAGNOSTIC_INNER_ONLY'
    assert route(dict(native_result='PASS'),b,0)=='ACCEPT_NATIVE_REQUESTED_SCOPE'
    for bad in (dict(max_plans=9),dict(max_generations=0),dict(max_diagnostic_replans=1),dict(attempts_per_plan=True)):
        with pytest.raises(ValueError):budget(bad)
    v=variants(dict(seed=7),b);assert v==[dict(seed=7,trajectory_strength=.55,trajectory_radius=12),
        dict(seed=7,trajectory_strength=.35,trajectory_radius=12)]
    assert variants(dict(seed=7,generator='adapter'),b)==[dict(seed=7),dict(seed=8)]

@pytest.mark.parametrize('field',['audio','features','feature_receipt','checkpoint','adapter','scaler','seed','trajectory_strength','overlap_policy'])
def test_cache_key_binds_all_execution_context(field):
    q=dict(audio='a',features='f',seed=1,output='one')
    before=execution_key(q,dict(sha256='plan'),'contract','code')
    changed=dict(q);changed[field]='changed'
    assert execution_key(changed,dict(sha256='plan'),'contract','code')!=before
    assert execution_key(dict(q,output='other'),dict(sha256='plan'),'contract','code')==before
    assert execution_key(q,dict(sha256='different'),'contract','code')!=before
    assert execution_key(q,dict(sha256='plan'),'new-contract','code')!=before

def test_cache_corruption_never_becomes_a_native_fail(tmp_path):
    feedback=tmp_path/'feedback.json';save(feedback,dict(native_result='FAIL'))
    record=dict(key='key',native_result='FAIL',
        artifacts={'feedback.json':dict(path=str(feedback),sha256=sha(feedback))})
    assert verify_cache(record,'key')['native_result']=='FAIL'
    feedback.write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):verify_cache(record,'key')

def test_timeout_reaps_owned_worker(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        launch([sys.executable,'-c','import time;time.sleep(30)'],tmp_path/'log',os.environ.copy(),.1)


def fake_controller(tmp_path,monkeypatch,result='FAIL',explode=False):
    import dunhuang.planning.replanner as r
    catalog,timing=fixture(tmp_path);parent=plan(catalog,timing,'audio')
    asset=tmp_path/'asset';asset.write_text('fixture')
    ref=dict(path=str(asset),sha256=sha(asset));calls=[]
    def launch_worker(argv,log,env,timeout):
        module=argv[2];args=argv[3:];Path(log).write_text('fixture')
        if module=='dunhuang.planning.feedback':
            out=Path(args[args.index('--out')+1])
            if '--contract-only' in args:save(out,dict(id='fixed-contract'))
            else:save(out,dict(native_result=result,broad_failure=True,associated_slots=[1],
                              recommended_route='MODEL_DIAGNOSIS',full_song_scope=False))
        elif module=='dunhuang.planning.worker':
            req=r.read(args[0]);out=Path(req['output'])
            selected=parent
            if 'replanning' in req:
                spec=req['replanning'];old=r.read(spec['parent']['path'])
                selected=propose_candidates(catalog,timing,'audio',
                    fixed_events={i:e for i,e in enumerate(old['events']) if i!=1},
                    excluded_events={1:[old['events'][1]['event_uid']]})[0]
                selected=revised_plan(selected,old,spec['feedback'],'REPLAN_ASSOCIATED_SLOTS',[1])
            save(out/'event_plan.json',selected);save(out/'event_plans.json',[selected])
        elif module=='dunhuang.pipeline':
            calls.append(module)
            if explode:raise subprocess.TimeoutExpired(argv,timeout)
            req=r.read(args[args.index('--config')+1]);out=Path(req['output']);out.mkdir()
            save(out/'request.json',req);selected=r.read(req['input_event_plan']['path'])
            save(out/'event_plan.json',selected)
            for name in ('physical.json','generation.json'):save(out/name,{})
            value=int(selected['events'][1]['event_uid'])+req['seed']+req.get('trajectory_strength',0)
            for name in ('normalized.npy','windows.npy','native.npy','native_fk.npy'):
                np.save(out/name,np.full((2,3),value))
        else:raise AssertionError(module)
    monkeypatch.setattr(r,'launch',launch_worker);monkeypatch.setattr(r,'code_identity',lambda:'fixed-code')
    q=dict(profile='dunhuang',seed=7,frames=270,output=str(tmp_path/'run'),edge_python='edge',native_python='native',
        **{key:ref for key in ('audio','features','feature_receipt','checkpoint','scaler','catalog')})
    return r,q,calls

def test_controller_broad_stop_resume_and_changed_native_contract(tmp_path,monkeypatch):
    r,q,calls=fake_controller(tmp_path,monkeypatch)
    q['closed_loop']=dict(enabled=True)
    out=r.run_closed_loop(q);state=r.read(out/'state.json')
    assert len(calls)==1 and state['stop_reason']=='STOP_MODEL_DIAGNOSIS'
    assert state['plans'][0]['native_result']=='UNKNOWN'
    assert not state['full_song_native_pass']
    r.run_closed_loop(dict(q,closed_loop=dict(q['closed_loop'],resume=True)))
    assert len(calls)==1
    save(out/'native_contract.json',dict(id='changed-contract'))
    with pytest.raises(ValueError,match='Native environment'):
        r.run_closed_loop(dict(q,closed_loop=dict(q['closed_loop'],resume=True)))

def test_controller_one_diagnostic_local_replan_then_budget_stop(tmp_path,monkeypatch):
    r,q,calls=fake_controller(tmp_path,monkeypatch)
    q['closed_loop']=dict(enabled=True,max_plans=2,max_generations=4,
        diagnostic_continue_on_broad_failure=True,max_diagnostic_replans=1)
    state=r.read(r.run_closed_loop(q)/'state.json')
    assert len(calls)==4 and len(state['plans'])==2 and state['diagnostic_replans']==1
    assert state['stop_reason']=='STOP_PLAN_BUDGET'
    assert [e['action'] for e in state['ledger']]==['INITIAL','REGENERATE_CONDITION_OR_SEED',
        'REPLAN_ASSOCIATED_SLOTS','REGENERATE_CONDITION_OR_SEED']
    assert state['event_global_blacklist']==[]

def test_controller_clip_pass_is_not_full_song_pass(tmp_path,monkeypatch):
    r,q,calls=fake_controller(tmp_path,monkeypatch,result='PASS')
    q['closed_loop']=dict(enabled=True)
    state=r.read(r.run_closed_loop(q)/'state.json')
    assert state['accepted'] and state['stop_reason']=='ACCEPT_NATIVE_CLIP_ONLY'
    assert len(calls)==1 and not state['full_song_native_pass'] and not state['full_generation_qualified']

def test_worker_timeout_stays_unknown_and_never_becomes_cached_fail(tmp_path,monkeypatch):
    r,q,calls=fake_controller(tmp_path,monkeypatch,explode=True)
    q['closed_loop']=dict(enabled=True)
    with pytest.raises(subprocess.TimeoutExpired):r.run_closed_loop(q)
    state=r.read(Path(q['output'])/'state.json')
    assert state['stop_reason']=='STOP_EXECUTION_UNKNOWN'
    assert state['ledger'][0]['native_result']=='UNKNOWN' and 'cache' not in state['ledger'][0]
    r.run_closed_loop(dict(q,closed_loop=dict(q['closed_loop'],resume=True)))
    assert len(calls)==1


def test_native_contract_survives_full_dataclass_json_roundtrip():
    from dunhuang.planning.feedback import contract_identity
    value=dict(config=dict(joints=(0,1,2),weights=(.1,.2)),limits=dict(jerk=1620.))
    sealed=contract_identity(value)
    assert sealed==json.loads(json.dumps(sealed))
    assert sealed==contract_identity(json.loads(json.dumps(value)))
    assert sealed!=contract_identity(dict(value,limits=dict(jerk=2000.)))
