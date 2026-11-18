"""Frozen two-stage execution: first B0/B3 effect, then remaining fixed controls."""
import subprocess
from dunhuang.research.warp_coupling_r53.population import Population
from dunhuang.research.whole_song_geometry.study import exact_layout,coverage
from .common import *
from .music import analyze,descriptors
from .slots import allocate,feasible_lengths,shuffled,centers
from .planning import source_kernels,build,choose,assemble
from .metrics import evaluate

def run(out,stage,render=True):
 out=fresh(out);pop=Population();units=read('outputs/event_coverage_r47_01/index/result.json')['units'];lookup={u['unit_uid']:u for u in units};bank=[lookup[i] for i in pop.old['source_bank']];assert len(bank)==CFG['source_units'];motion,metric_motion=source_kernels(pop,bank);results=[];arrays=dict(J=pop.net.joints,parents=pop.net.parents,motion_objective_similarity=motion,motion_evaluation_similarity=metric_motion);graphs={};music_meta={}
 for song in ([CFG['main_song']] if stage=='main' else CFG['songs']):
  track=pop.tracks[song];music,meta=analyze(track['path']);frames=meta['frames'];assert abs(frames-track['structure']['frames'])<=1;music_meta[str(song)]=meta
  for name,value in music.items():arrays[f'music{song}_{name}']=value
  fixed=exact_layout(frames);assert sum(r['kind']=='core' for r in fixed)==CFG['events'];aware,slot_score=allocate(frames,music['boundary_reward'],feasible_lengths(bank));permuted,shuffle=shuffled(aware,song);layouts=dict(B0=fixed,B1=aware,B2=permuted,B3=aware)
  graph_cache={};arms=['B0','B3'] if stage=='main' else CFG['arms']
  for arm in arms:
   spans=layouts[arm];covered=coverage(spans,frames);key=identity(spans)
   if key not in graph_cache:
    pools,edges,ledger=build(pop,bank,spans,track);graph_cache[key]=(pools,edges,ledger);graphs[f'{song}_{key}']=dict(song=song,spans=spans,**ledger);print('R56_GRAPH',song,arm,ledger['geometry_trials'],ledger['eligible'],flush=True)
   pools,edges,ledger=graph_cache[key];sim=kernel(descriptors(music['chroma'],spans));choice,layers=choose(pools,edges,pop.reference,sim if arm=='B3' else None,motion if arm=='B3' else None);traj,plan=assemble(pop,pools,edges,choice,spans);record=dict(song=song,arm=arm,frames=frames,regions=spans,boundary_centers=centers(spans),graph_key=f'{song}_{key}',search_layers=layers,slot_objective=slot_score if arm!='B0' else None,shuffle=shuffle if arm=='B2' else None,**plan)
   if traj is not None:
    for name,value in traj.items():arrays[f'song{song}_{arm}_{name}']=value
    record['metrics']=evaluate(pop,traj,plan,spans,music,metric_motion);record['rotation_sha256']=array_hash(traj['R']);record['positions_sha256']=array_hash(traj['P'])
    if song==0 and arm=='B0':
     prior=read('outputs/whole_song_geometry_02/main/result.json');old=np.load('outputs/whole_song_geometry_02/main/arrays.npz');assert np.array_equal(traj['R'],old['rotations']) and np.array_equal(traj['P'],old['positions_root_relative']);assert [e['event_uid'] for e in plan['events']]==[e['event_uid'] for e in prior['selected']];record['frozen_baseline_exact']=True
   results.append(record);print('R56_PLAN',song,arm,plan['status'],flush=True)
  if stage=='controls':
   aa=next(r for r in results if r['song']==song and r['arm']=='B1');bb=next(r for r in results if r['song']==song and r['arm']=='B3');assert aa['graph_key']==bb['graph_key']
 result=dict(stage=stage,records=results,music=music_meta,graphs=graphs,binding=dict(parent=pop.binding,science=binding(),protocol_sha256=sha(Path(__file__).with_name('protocol.json')),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),baseline_manifest_sha256=sha('outputs/whole_song_geometry_02/artifacts_manifest.json')),world_root_contact='UNKNOWN',phrase_truth='NOT_EVALUABLE',human_reviewed=False,can_route=False,training_steps=0,optimizer_calls=0)
 if stage=='controls':
  prior=read(ROOT/'main/result.json');old=np.load(ROOT/'main/arrays.npz')
  for key,value in arrays.items():
   if key.startswith(('song0_B0_','song0_B3_')):assert np.array_equal(value,old[key],equal_nan=True),key
  for r in prior['records']:
   now=next(x for x in results if x['song']==r['song'] and x['arm']==r['arm']);assert now==r
 save_arrays(out/'arrays.npz',arrays);write(out/'result.json',result)
 if render:
  from .view import viewer
  viewer(out,pop,arrays,result)
 return result

def verify(out,source):
 out=fresh(out);source=Path(source);expected=read(source/'result.json');assert expected['binding']['science']==binding()
 from dunhuang.research.coverage_r34.common import forbid_optimizers
 forbid_optimizers();actual=run(out/'rebuild',expected['stage'],render=False);source_commit=expected['binding']['code_commit'];replay_commit=actual['binding']['code_commit']
 for name,digest in expected['binding']['science'].items():
  data=subprocess.check_output(['git','show',source_commit+':src/dunhuang/research/music_structure_r56/'+name]);assert __import__('hashlib').sha256(data).hexdigest()==digest
 actual['binding']['code_commit']=source_commit;assert actual==expected;assert sha(source/'arrays.npz')==sha(out/'rebuild/arrays.npz');write(out/'result.json',dict(exact_result=True,exact_arrays_bytes=True,source_execution_commit=source_commit,replay_execution_commit=replay_commit,only_ignored_runtime_field='binding.code_commit',source_sha256=sha(source/'result.json'),arrays_sha256=sha(source/'arrays.npz'),optimizer_calls=0,model_calls=0,can_route=False))
