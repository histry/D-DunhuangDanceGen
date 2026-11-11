"""Bounded orchestration only; Native remains the read-only authority."""
import copy,json,os,signal,subprocess,time,sys,uuid
from pathlib import Path
import numpy as np
from dunhuang.io import checked,sha
from .feedback import digest
from .event_plan import semantic_plan_id

def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
    temporary.replace(path)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def reference(path):
    return dict(path=str(Path(path).resolve()),sha256=sha(path))

def budget(config):
    defaults=dict(max_plans=3,attempts_per_plan=2,max_generations=6,
        max_active_seconds=3600,candidate_timeout_seconds=600,
        diagnostic_continue_on_broad_failure=False,max_diagnostic_replans=0,
        inner_order='condition_first',low_response_rms=1e-3)
    value=dict(defaults,**config)
    for name,upper in [('max_plans',8),('attempts_per_plan',2),('max_generations',16),
                       ('max_active_seconds',86400),('candidate_timeout_seconds',3600)]:
        if type(value[name]) is not int or not 1<=value[name]<=upper:
            raise ValueError('Invalid bounded budget: '+name)
    if type(value['max_diagnostic_replans']) is not int or not 0<=value['max_diagnostic_replans']<=1:
        raise ValueError('At most one explicit diagnostic replan')
    if type(value['diagnostic_continue_on_broad_failure']) is not bool:
        raise ValueError('Diagnostic override must be boolean')
    if value['max_diagnostic_replans'] and not value['diagnostic_continue_on_broad_failure']:
        raise ValueError('Diagnostic replans require explicit override')
    if value['inner_order'] not in ('condition_first','seed_first'):
        raise ValueError('Unknown registered inner-loop order')
    if not 0<=value['low_response_rms']<=.01:raise ValueError('Invalid response diagnostic threshold')
    return value

def variants(q,b):
    first=dict(seed=q['seed'])
    if q.get('generator','temporal')=='temporal':
        first.update(trajectory_strength=q.get('trajectory_strength',.55),
                     trajectory_radius=q.get('trajectory_radius',12))
    second=dict(first)
    if b['inner_order']=='condition_first' and q.get('generator','temporal')=='temporal':
        second['trajectory_strength']=.35 if first['trajectory_strength']!=.35 else .55
    else:second['seed']=q['seed']+1
    return [first,second][:b['attempts_per_plan']]

def execution_key(q,plan,contract,code):
    # Conservative: bind every caller-supplied input/knob, full plan file and
    # environment, not merely an event ID. Paths may cause safe cache misses.
    payload={k:v for k,v in q.items() if k not in ('output','closed_loop','input_event_plan')}
    return digest(dict(schema='candidate_identity_v1',request=payload,plan=plan,
                       native_contract=contract,code=code,sampler_steps=50,window=150,stride=75))

def verify_cache(record,key):
    if record['key']!=key:raise ValueError('Cache key mismatch')
    for item in record['artifacts'].values():checked(item)
    feedback=read(checked(record['artifacts']['feedback.json']))
    if feedback['native_result']!=record['native_result']:raise ValueError('Cache Native state mismatch')
    return feedback

def route(feedback,b,diagnostic_replans):
    if feedback['native_result']=='PASS':return 'ACCEPT_NATIVE_REQUESTED_SCOPE'
    if feedback['broad_failure']:
        if not b['diagnostic_continue_on_broad_failure']:return 'STOP_MODEL_DIAGNOSIS'
        if diagnostic_replans>=b['max_diagnostic_replans']:return 'DIAGNOSTIC_INNER_ONLY'
        return 'DIAGNOSTIC_INNER_AND_ONE_REPLAN'
    return feedback['recommended_route']

def launch(argv,log,env,timeout):
    """Kill only the subprocess group we created, including its CUDA child."""
    if timeout<=0:raise TimeoutError('Active wall-clock budget exhausted')
    with Path(log).open('w') as stream:
        process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,
            start_new_session=os.name!='nt')
        try:code=process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
            else:process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
                else:process.kill()
                process.wait()
            raise
        if code:raise RuntimeError('Worker exited '+str(code)+'; see '+str(log))

def code_identity():
    root=Path(__file__).resolve().parents[3]
    files=sorted((root/'src'/'dunhuang').rglob('*.py'))
    import importlib.metadata as metadata
    versions={}
    for package in ('numpy','scipy','torch'):
        try:versions[package]=metadata.version(package)
        except metadata.PackageNotFoundError:versions[package]=None
    return dict(source=digest({str(p.relative_to(root)):sha(p) for p in files}),
                python=sys.version,runtime=versions)

def run_closed_loop(request):
    q=copy.deepcopy(request);cfg=q.pop('closed_loop');resume=cfg.pop('resume',False)
    b=budget(cfg)
    if q.get('profile')!='dunhuang' or q.get('retrieval_policy','heuristic')!='heuristic' or q.get('learned_retrieval'):
        raise ValueError('V0.8 closed loop requires the explicit heuristic Dunhuang path')
    if q.get('generator','temporal') not in ('temporal','adapter'):raise ValueError('Unsupported generator')
    if type(q.get('seed')) is not int:raise ValueError('Registered seed required')
    for name in ('audio','features','feature_receipt','checkpoint','scaler','catalog'):
        checked(q[name])
    if q.get('adapter'):checked(q['adapter'])
    code=code_identity();identity=digest(dict(request=q,budget=b,code=code))
    root=Path(q['output']).resolve();state_path=root/'state.json'
    env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[2]);env.pop('PYTHONHOME',None)
    if root.exists():
        if not resume:raise FileExistsError('Explicit --resume required: '+str(root))
        state=read(state_path)
        if state['input_identity']!=identity:raise ValueError('Resume inputs/code/budget changed')
        # Audit receipts before reusing any result. A partially written worker
        # output is UNKNOWN, never a cached FAIL or automatic fresh retry.
        for entry in state['ledger']:
            if entry.get('cache'):verify_cache(read(checked(entry['cache'])),entry['key'])
        if not state.get('terminal'):
            state['stop_reason']='STOP_INTERRUPTED_UNKNOWN_REQUIRES_NEW_RUN'
            state['terminal']=True
            save(state_path,state)
        save(root/('resume_'+uuid.uuid4().hex+'.json'),dict(state_sha256=sha(state_path),
             entries_verified=len(state['ledger']),new_generations=0,terminal=state['stop_reason']))
        return root
    root.mkdir(parents=True);(root/'requests').mkdir();(root/'cache').mkdir()
    state=dict(schema='native_closed_loop_v1',input_identity=identity,code=code,budget=b,
        ledger=[],plans=[],generations_started=0,diagnostic_replans=0,active_seconds=0.,
        terminal=False,accepted=None,full_song_native_pass=False,
        full_generation_qualified=False,event_global_blacklist=[],learned_risk=None)
    save(root/'request.json',dict(q,closed_loop=dict(b,enabled=True)))
    start=time.monotonic()
    def persist():
        state['active_seconds']=time.monotonic()-start;save(state_path,state)
    def remaining():
        return b['max_active_seconds']-(time.monotonic()-start)
    def worker(python,module,args,label):
        launch([python,'-m',module,*map(str,args)],root/(label+'.log'),env,
               min(b['candidate_timeout_seconds'],remaining()))
    def planning(parent=None,feedback=None):
        folder=root/('planning_'+str(len(state['plans'])));folder.mkdir()
        req=dict(q,output=str(folder),plan_count=min(3,b['max_plans']))
        if parent:
            req.pop('input_event_plan',None)
            slots=feedback['associated_slots'][:1]
            if not slots:return None
            req['replanning']=dict(parent=reference(parent),changed_slots=slots,
                                    feedback=reference(last_feedback))
        path=folder/'request.json';save(path,req)
        worker(q['native_python'],'dunhuang.planning.worker',[path],folder.name)
        proposals=read(folder/'event_plans.json') if (folder/'event_plans.json').exists() else [read(folder/'event_plan.json')]
        known={p['semantic_id'] for p in state['plans']}
        for proposal in proposals:
            sid=semantic_plan_id(proposal)
            if sid in known:continue
            path=folder/'selected.json';save(path,proposal)
            state['plans'].append(dict(plan=reference(path),semantic_id=sid,native_result='UNKNOWN',
                parent=reference(parent) if parent else None))
            persist();return path
        return None
    persist()
    try:
        contract_path=root/'native_contract.json'
        worker(q['native_python'],'dunhuang.planning.feedback',
               ['--request',root/'request.json','--out',contract_path,'--contract-only'],'contract')
        contract=read(contract_path)
        plan=planning()
        stop=None;previous_variant_results={}
        while plan and not stop:
            feedback=None
            for vi,variant in enumerate(variants(q,b)):
                if state['generations_started']>=b['max_generations']:
                    stop='STOP_GENERATION_BUDGET';break
                if remaining()<=0:stop='STOP_ACTIVE_TIME_BUDGET';break
                candidate=dict(q,**variant,input_event_plan=reference(plan))
                case=root/('candidate_'+str(len(state['ledger'])))
                candidate['output']=str(case)
                key=execution_key(candidate,reference(plan),contract,code)
                cache_path=root/'cache'/(key+'.json')
                entry=dict(key=key,plan_id=semantic_plan_id(read(plan)),variant=variant,
                    native_result='UNKNOWN',case=str(case),action='INITIAL' if not state['ledger'] else
                    ('REGENERATE_CONDITION_OR_SEED' if vi else 'REPLAN_ASSOCIATED_SLOTS'))
                state['ledger'].append(entry);persist()
                if cache_path.exists():
                    feedback=verify_cache(read(cache_path),key);entry['cache_hit']=True
                    case=Path(read(cache_path)['case'])
                else:
                    config=root/'requests'/('candidate_'+str(len(state['ledger'])-1)+'.json');save(config,candidate)
                    state['generations_started']+=1;persist()
                    worker(q['edge_python'],'dunhuang.pipeline',['--config',config],case.name)
                    last_feedback=case/'feedback.json'
                    worker(q['native_python'],'dunhuang.planning.feedback',[
                        '--request',case/'request.json','--case',case,'--contract-file',contract_path,
                        '--out',last_feedback],case.name+'_feedback')
                    feedback=read(last_feedback)
                    artifacts={name:reference(case/name) for name in (
                        'request.json','event_plan.json','native.npy','native_fk.npy','windows.npy',
                        'normalized.npy','physical.json','generation.json','feedback.json')}
                    save(cache_path,dict(key=key,case=str(case),artifacts=artifacts,native_result=feedback['native_result']))
                entry['cache']=reference(cache_path);last_feedback=case/'feedback.json'
                entry.update(native_result=feedback['native_result'],feedback=reference(last_feedback),
                             route=route(feedback,b,state['diagnostic_replans']))
                state['plans'][-1]['last_generated_native_result']=feedback['native_result']
                persist()
                if feedback['native_result']=='PASS':
                    state['accepted']=entry['cache'];state['full_song_native_pass']=feedback['full_song_scope']
                    stop='ACCEPT_NATIVE_FULL_SONG' if feedback['full_song_scope'] else 'ACCEPT_NATIVE_CLIP_ONLY';break
                # A matched seed/condition contrast is a response diagnostic,
                # never a probability of physical success or event feasibility.
                signature=digest(variant)
                if signature in previous_variant_results:
                    previous=previous_variant_results[signature]
                    old=read(previous/'event_plan.json');new=read(case/'event_plan.json')
                    if [e['event_uid'] for e in old['events']]!=[e['event_uid'] for e in new['events']]:
                        left=np.load(previous/'normalized.npy',allow_pickle=False)
                        right=np.load(case/'normalized.npy',allow_pickle=False)
                        rms=float(np.sqrt(np.mean((left.astype(float)-right.astype(float))**2)))
                        entry['matched_event_response_rms']=rms
                        if rms<=b['low_response_rms']:stop='STOP_LOW_RESPONSE_FOR_TESTED_PAIR';persist();break
                previous_variant_results[signature]=case
                if entry['route']=='STOP_MODEL_DIAGNOSIS':stop=entry['route'];break
                if entry['route']=='BOUNDARY_CONTEXT_DIAGNOSIS':stop='STOP_BOUNDARY_CONTEXT_NOT_IMPLEMENTED';break
            if stop:break
            if len(state['plans'])>=b['max_plans']:stop='STOP_PLAN_BUDGET';break
            if feedback is None:stop='STOP_NO_FEEDBACK';break
            if feedback['broad_failure']:
                if state['diagnostic_replans']>=b['max_diagnostic_replans']:
                    stop='STOP_MODEL_DIAGNOSIS_DIAGNOSTIC_BUDGET';break
                state['diagnostic_replans']+=1
                state.setdefault('diagnostic_overrides',[]).append(dict(feedback=reference(last_feedback),
                    reason='Explicit bounded broad-failure interface diagnostic, not a quality intervention'))
            next_plan=planning(plan,feedback)
            if next_plan is None:stop='STOP_NO_NOVEL_LOCAL_PLAN';break
            plan=next_plan
        state['stop_reason']=stop or 'STOP_NO_PLAN'
    except Exception as error:
        import traceback
        state['stop_reason']='STOP_EXECUTION_UNKNOWN'
        state['error']=repr(error);state['traceback']=traceback.format_exc()
        if state['ledger'] and state['ledger'][-1]['native_result']=='UNKNOWN':
            state['ledger'][-1]['execution_error']=repr(error)
        state['terminal']=True;persist()
        raise
    state['terminal']=True;persist()
    return root
