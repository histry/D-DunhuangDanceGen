"""Actual fixed-source executions, frozen old selections, and new DP/DTW.

No model scoring or training is performed. Saved historical geometry is reused
only for its exact old execution; newly proposed clocks rebuild original G1.
"""
from copy import deepcopy
from collections import Counter
from pathlib import Path
import secrets,sys,time
import numpy as np
import scipy
from dunhuang.research.complete_r36.core import rotation_sample
from dunhuang.research.warp_coupling_r53.common import array_hash,save_arrays
from dunhuang.research.warp_coupling_r53.warp_lattice import validate
from dunhuang.research.sdmr_warp_rank import connectivity
from dunhuang.research.paired_event_r41.events import profiles
from dunhuang.research.rhythm_r66 import metrics as body_metrics
from dunhuang.research.local_timing_r68.study import pair_metrics,music_proxy
from dunhuang.research.local_timing_r68.warp import uniform as frozen_uniform
from .common import CFG,ROOT,read,write,sha,fresh,binding,seal,verify_manifest
from . import solver

R67=Path('outputs/sdmr_warp_rank_01/main')
R68=Path('outputs/local_timing_r68_01/controls')

def verify_execution(case,q,R,connection,full_R,source):
 q=np.asarray(q,float);R=np.asarray(R,float)
 a,b=case['source_interval']
 if q.shape!=(240,) or R.shape!=(240,22,3,3) or not np.isfinite(q).all() or not (np.diff(q)>0).all():raise ValueError('Actual finite complete monotone240 clock required')
 if q[0]!=a or q[-1]!=b-1 or np.floor(q).min()<a or np.ceil(q).max()>=b:raise ValueError('Source support or endpoints changed')
 actual=rotation_sample(source,q)
 if array_hash(actual)!=array_hash(R):raise ValueError('Saved rotations do not reproduce actual original source reads')
 if connection.get('candidate_R_sha256')!=array_hash(R):raise ValueError('G1 receipt is for another execution')
 if connection.get('parent_record_key')!=case['parent_record_key'] or connection.get('gap_frames')!=48:raise ValueError('Fixed neighbor or transition budget changed')
 if full_R is not None:
  if np.asarray(full_R).shape!=(528,22,3,3) or array_hash(full_R)!=connection.get('full_R_sha256'):raise ValueError('Full pair receipt mismatch')
  core=full_R[:240] if case['parent_side']=='left' else full_R[-240:]
  if not np.array_equal(core,R):raise ValueError('G1 edited the source Core')
  if array_hash(full_R[240:288])!=connection.get('bridge_R_sha256'):raise ValueError('Bridge receipt mismatch')
 return True

def raw_ties(rows,field):
 values=[r['scores'].get(field) for r in rows]
 if any(v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v)) for v in values):raise ValueError('Nonfinite or nonnumeric score')
 good=[(i,v) for i,v in enumerate(values) if v is not None]
 if not good:return []
 top=max(v for i,v in good)
 return [i for i,v in good if abs(v-top)<=1e-12]

def original_selection(case,method,bank,records,arrays,root=R67):
 rows=[records[(case['key'],c['uid'])] for c in bank['candidates']]
 ties=[0] if method=='E1W' else raw_ties(rows,method)
 sealed=read(root/'selections'/(case['key']+'.json'))
 if method!='E1W' and ties!=sealed['raw_score_ties'][method]:raise ValueError('Historical raw maximum tie set changed')
 if not ties:return None
 i=ties[0];candidate=bank['candidates'][i];row=rows[i]
 key=case['key']+'_'+candidate['uid'][:16]
 R=arrays[key];q=np.asarray(candidate['q'],float)
 if array_hash(R)!=row['rotation_sha256'] or array_hash(q)!=row['q_sha256'] or array_hash(q)!=candidate['q_sha256']:raise ValueError('Saved R67 candidate identity mismatch')
 fullkey=key+'_G1_full_R';full=arrays[fullkey] if fullkey in arrays.files else None
 return dict(q=q,R=R,full_R=full,connection=deepcopy(row['connectivity']),warp=candidate['time_warp'],
  selection=dict(origin='EXACT_R67_RAW_SELECTION_REUSE',candidate_uid=candidate['uid'],candidate_index=i,
   actual_candidate_count=len(rows),tie_indices=ties,display_tie_rule='FIRST_REGISTERED_NOT_POSTHOC_BEST',
   score=row['scores'].get(method),historical_encoding_calls=len(rows),historical_geometry_calls=len(rows),
   current_encoding_calls=0,current_geometry_calls=0,historical_wall_seconds=None))

def original_uniform(case,records,arrays):
 old=records[(case['key'],'Uniform')]
 if old['status']!='LEGAL':return None
 key=case['key']+'__Uniform__';R=arrays[key+'R'];q=arrays[key+'q']
 if array_hash(R)!=old['R_sha256'] or array_hash(q)!=old['q_sha256']:raise ValueError('Saved R68 Uniform identity mismatch')
 expected=frozen_uniform(case['source_interval'],case['target_interval'])['q']
 if not np.array_equal(q,expected):raise ValueError('Uniform clock no longer uniform')
 keyfull=key+'G1_full_R';full=arrays[keyfull] if keyfull in arrays.files else None
 return dict(q=q,R=R,full_R=full,connection=deepcopy(old['connection']),warp=old['warp']['time_warp'],
  selection=dict(origin='EXACT_R68_UNIFORM_REUSE',actual_candidate_count=1,current_encoding_calls=0,current_geometry_calls=0,historical_geometry_calls=1,historical_wall_seconds=None))

def proposed(case,method,source,clock,ctx):
 settings=CFG['solver'];a,b=case['source_interval'];A,B=case['target_interval']
 if method=='BeatDP':
  events=[p for p in case['phase_inputs'] if p['type']=='upper_speed_peak' and a<=p['evidence_span'][0]<=p['frame']<p['evidence_span'][1]<=b]
  result=solver.beat_dp(case['source_interval'],case['target_interval'],[p['frame'] for p in events],
   [p['score'] for p in events],clock['beat_frames'],**settings)
 else:
  result=solver.rhythm_dtw(case['source_interval'],case['target_interval'],profiles(source)['upper'][a:b],
   clock['onset_strength'][A:B],**{k:v for k,v in settings.items() if k!='sigma_frames'})
 result=deepcopy(result);runtime=result.pop('runtime_seconds',None)
 if result['status']!='LEGAL':return dict(result=result,R=None,q=None,full_R=None,connection=None),runtime
 q=np.asarray(result.pop('source_positions'),float);result.pop('q',None)
 validate(result['time_warp'],case['source_interval'],case['target_interval'])
 R=rotation_sample(source,q)
 # Every new execution gets a real new solve, even when it equals another arm.
 ctx['cache'].clear();connection,geometry=connectivity.check(case,R,ctx)
 return dict(result=result,q=q,R=R,full_R=geometry.get('full_R'),connection=connection,
  warp=result['time_warp'],selection=dict(origin='NEW_REGISTERED_BASELINE',current_encoding_calls=0,current_geometry_calls=1)),runtime

def summary(records):
 methods={}
 for name in CFG['methods']:
  rows=[r for r in records if r['method']==name]
  methods[name]=dict(registered_cases=len(rows),statuses=dict(Counter(r['status'] for r in rows)),
   full_pair_available=sum(r.get('full_R_sha256') is not None for r in rows),
   original_G1_screen_passed=sum((r.get('connection') or {}).get('passes_original_screen') is True for r in rows),
   current_geometry_calls=sum(r.get('selection',{}).get('current_geometry_calls',0) for r in rows),
   current_encoding_calls=0,quality_winner=None)
 return dict(registered_cases=12,source_intervals=8,music_tracks=3,methods=methods,
  registered_method_positions=len(records),distinct_saved_full_trajectories=len({r['full_R_sha256'] for r in records if r.get('full_R_sha256')}),
  distinct_hash_count_scope='EXACT_ARRAY_BYTES_NOT_INDEPENDENT_SAMPLES_OR_SEMANTICALLY_DIFFERENT_CLOCKS',
  human_review_records=0,artistic_quality='NOT_EVALUABLE',Native24='NOT_EVALUABLE',can_route=False,
  scope='RETROSPECTIVE_EXPOSED_FIXED_SOURCE_BASELINE_COMPARISON_NOT_EQUAL_DOMAIN_OR_COMPUTE')

def run(out=None,*,blind_seed=None):
 target=fresh(out or ROOT/'main');start=time.perf_counter()
 old=verify_manifest(R67,CFG['r67_main_manifest_sha256']);old68=verify_manifest(R68,CFG['r68_controls_manifest_sha256'])
 pop=read(R67/'population.json');cases=deepcopy(pop['cases']);ctx=connectivity.prepare()
 if len(cases)!=12 or len({(c['source_uid'],tuple(c['source_interval'])) for c in cases})!=8:raise ValueError('Fixed twelve/eight population changed')
 banks={r['case_key']:r['bank'] for r in read(R67/'banks.json')}
 oldrows={(r['case_key'],r['candidate_uid']):r for r in read(R67/'records.json')}
 old68rows={(r['case_key'],r['method']):r for r in read(R68/'records.json')}
 arrays67=np.load(R67/'arrays.npz',allow_pickle=False);arrays68=np.load(R68/'arrays.npz',allow_pickle=False)
 cp=Path('outputs/timing_r64_01/main/music_clocks.json')
 if sha(cp)!=CFG['r64_clocks_sha256']:raise ValueError('Frozen actual music clock changed')
 clocks=read(cp);display=[];records=[];arrays={};timings=[]
 for case in cases:
  if case['key'] not in ctx['registered']:raise ValueError('Unregistered original G1 pair')
  for field in ('source_uid','source_interval','target_interval','parent_side','parent_case_key','time_warp'):
   if case[field]!=ctx['registered'][case['key']][field]:raise ValueError('Parent population mismatch')
  source=ctx['lib'].rotations[case['source_uid']];clock=clocks[str(case['songidx'])]
  if clock['audio_sha256']!=case['audio_sha256']:raise ValueError('Audio identity mismatch')
  case['pair_music_interval']=ctx['pairs'][case['parent_case_key']]['case']['music_interval']
  panels=[]
  for method in CFG['methods']:
   tick=time.perf_counter();solver_seconds=None
   if method=='Uniform':v=original_uniform(case,old68rows,arrays68)
   elif method in ('E1W','Single64','Multi64'):v=original_selection(case,method,banks[case['key']],oldrows,arrays67)
   else:v,solver_seconds=proposed(case,method,source,clock,ctx)
   status='NOT_EVALUABLE' if v is None else v.get('result',{}).get('status','LEGAL')
   row=dict(case_key=case['key'],method=method,status=status,reason='NO_SAVED_SELECTION' if v is None else v.get('result',{}).get('reason'),
    source_uid=case['source_uid'],source_interval=case['source_interval'],target_interval=case['target_interval'],
    pair_music_interval=case['pair_music_interval'],audio_sha256=case['audio_sha256'],human_reviewed=False,
    semantic_phase_truth='UNKNOWN',artistic_quality='NOT_EVALUABLE',Native24='NOT_EVALUABLE',can_route=False)
   if v is not None:
    row.update(selection=v.get('selection',{}),solver=v.get('result'),connection=v.get('connection'))
   if v is not None and v.get('R') is not None:
    q,R,full=v['q'],v['R'],v['full_R'];verify_execution(case,q,R,v['connection'],full,source)
    neighbor='right' if case['parent_side']=='left' else 'left'
    expected_neighbor=ctx['pairs'][case['parent_case_key']][neighbor]
    if v['connection'].get('numerical_screen_reference')!=ctx['reference'] or v['connection'].get('neighbor_R_sha256')!=array_hash(expected_neighbor):raise ValueError('Historical G1 used another frozen neighbor or screen')
    validate(v['warp'],case['source_interval'],case['target_interval'])
    row.update(time_warp=v['warp'],R_sha256=array_hash(R),q_sha256=array_hash(q),full_R_sha256=array_hash(full) if full is not None else None,
     metrics=body_metrics.measure(R,source,ctx['J'],ctx['parents'],q,case['source_interval'],include_curves=False),
     pair_dynamics=pair_metrics(full,ctx['J'],ctx['parents']) if full is not None else None,
     proxy=music_proxy(R,q,case,clock))
    q0=np.asarray(banks[case['key']]['candidates'][0]['q'],float)
    row['source_clock_change_from_E1W']=dict(mean_absolute_frames=float(np.mean(abs(q-q0))),maximum_absolute_frames=float(np.max(abs(q-q0))),same_under_registered_R67_atol1e9=bool(np.allclose(q,q0,atol=1e-9,rtol=0)))
    key=case['key']+'__'+method;arrays[key+'__R']=R;arrays[key+'__q']=q
    if full is not None:arrays[key+'__full_R']=full
    panels.append(dict(method=method,R=R,q=q,full_R=full,row=row))
   else:panels.append(dict(method=method,R=None,q=None,full_R=None,row=row))
   records.append(row);timings.append(dict(case_key=case['key'],method=method,current_wall_seconds=time.perf_counter()-tick,solver_wall_seconds=solver_seconds,
    includes_parent_hashing=False,historical_R67_encoding_and_search_seconds=None,not_equal_compute=True))
  a,b=case['source_interval'];display.append((case,panels,dict(R=source[a:b],fps=30)))
  print('R69_CASE',len(display),12,flush=True)
 arrays67.close();arrays68.close();save_arrays(target/'arrays.npz',arrays)
 write(target/'population.json',dict(cases=cases,source_intervals=8,music_tracks=3,scope=CFG['population']))
 write(target/'records.json',records);write(target/'summary.json',summary(records))
 write(target/'registration.json',dict(binding=binding(),protocol=CFG,parent_manifests=dict(r67=sha(R67/'manifest.json'),r68=sha(R68/'manifest.json')),
  runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),new_geometry_counters=ctx['counters'],new_encoder_calls=0))
 seed=secrets.token_bytes(32) if blind_seed is None else blind_seed
 (target/'admin').mkdir()
 (target/'admin/seed.hex').write_bytes(seed.hex().encode('ascii')+b'\n')
 from .viewer import build
 build(target,display,ctx['J'],ctx['parents'],clocks,blind_seed=seed)
 from .reviews import summarize_reviews
 write(target/'quality_state.json',summarize_reviews(read(target/'admin/map.json'),[]))
 write(target/'runtime.json',dict(rows=timings,total_current_wall_seconds=time.perf_counter()-start,scope='CURRENT_REUSE_READS_AND_NEW_SOLVES_NOT_FULL_R67_RECOMPUTATION'))
 seal(target);print('R69_FINISHED',len(records),flush=True)
 return target

def verify(out):
 source=ROOT/'main';manifest=verify_manifest(source);current=binding()
 if manifest['binding']['code']!=current['code'] or manifest['binding']['dependencies']!=current['dependencies']:raise ValueError('Scientific code changed')
 seed=bytes.fromhex((source/'admin/seed.hex').read_text().strip())
 target=run(out,blind_seed=seed)
 differences=[n for n,h in manifest['files'].items() if not (target/n).is_file() or sha(target/n)!=h]
 write(target/'independent.json',dict(scientific_outputs_exact=not differences,differences=differences,rebuilt_files=len(manifest['files']),
  source_manifest_sha256=sha(source/'manifest.json'),binding=current,fresh_process=True,new_neural_training_steps=0))
 if differences:raise ValueError('Independent replay differed')
