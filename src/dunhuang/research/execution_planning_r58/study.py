from .common import *
from .profiles import Assets,uniform_profiles
from .graph import build,load,choose,path_capacity
from dunhuang.research.music_structure_r56.planning import source_kernels
from dunhuang.research.music_structure_r56.metrics import evaluate

def assemble(assets,graph,data,choice):
 if choice is None:return None,dict(status='NO_PATH_IN_REGISTERED_FINITE_SEARCH')
 lookup={n['id']:n for layer in graph['layers'] for n in layer};edges={(e['left'],e['right']):e for e in graph['edges']};nodes=[lookup[key] for key in choice['ids']];spans=regions_for(nodes);N=spans[-1]['interval'][1];R=[];events=[];connections=[];source_frame=np.full(N,np.nan);source_event=np.full(N,-1,np.int32)
 for k,n in enumerate(nodes):
  a,b=n['target'];R.append(data[n['id']]);w=np.asarray(n['warp']['time_warp']);q=np.interp(np.arange(a,b),w[:,1],w[:,0]);source_frame[a:b]=q;source_event[a:b]=k;unit=next(u for u in assets.bank if u['unit_uid']==n['event_uid']);events.append(dict(event_uid=n['event_uid'],source_uid=n['source_uid'],source_interval=n['source_interval'],target_interval=n['target'],time_warp=n['warp'],bank=next(i for i,u in enumerate(assets.bank) if u['unit_uid']==n['event_uid']),phase_evidence_truncated_count=unit['stage_evidence_truncated'],execution_profile={key:n[key] for key in ('activity','metrics','native','source_normalized','execution_cost','boundary_state','native_sha256')}))
  if k<len(nodes)-1:
   e=edges[n['id'],nodes[k+1]['id']];assert e['eligible'];R.append(data['edge_'+e['key']]);connections.append(e['evidence'])
 R=np.concatenate(R);assert len(R)==N and np.isfinite(R).all();P=relative_fk(R,assets.pop.net.joints,assets.pop.net.parents).astype('<f4');arrays=dict(R=R,P=P,source_frame=source_frame,source_event=source_event,world_root_translation=np.full((N,3),np.nan),world_root_observed=np.zeros(N,bool));return arrays,dict(status='NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW',events=events,connections=connections,regions=spans,score=choice['score'],unit_histogram=choice['units'],source_histogram=choice['sources'],P0_time_coverage=True,P1_selected_connections=True,world_root_contact='UNKNOWN',native24='NOT_EVALUABLE')

def music_values(assets,song):return {name:assets.z[f'music{song}_{name}'] for name in ('spec','chroma','boundary_reward','proposal','held_boundary','proxy_boundaries','proxy_accents')}
def baseline(assets,song,arm,arrays):
 r=assets.record(song,arm).copy()
 for name in ('R','P','source_frame','source_event','world_root_translation','world_root_observed'):arrays[f'song{song}_{arm}_{name}']=assets.z[f'song{song}_{arm}_{name}']
 r['R56_frozen_exact']=True;return r
def median_ratio(profiles,order):
 values=[p['source_normalized'][order]['ratio'] for p in profiles if p['source_normalized'][order]['ratio'] is not None];return float(np.median(values)) if values else None
def summarize_selected(record):
 profiles=[e['execution_profile'] for e in record['events']];return dict(mean_execution_distortion=float(np.mean([p['execution_cost'] for p in profiles])),mean_activity=float(np.mean([p['activity'] for p in profiles])),median_A2=median_ratio(profiles,'2'),median_A3=median_ratio(profiles,'3'),scope='OPTIMIZED_PROFILE_READOUT_NOT_INDEPENDENT_QUALITY_TRUTH')

def run(out,stage,render=True,rebuild_main=False):
 if stage=='controls':assert read(ROOT/'receipts/main_browser.json')['passed']
 out=fresh(out);assets=Assets();arrays=dict(J=assets.pop.net.joints,parents=assets.pop.net.parents);records=[];graphs={};_,motion_fk=source_kernels(assets.pop,assets.bank);timing={};music={}
 if stage=='main':write(out/'uniform_profiles.json',uniform_profiles(assets))
 for song in ([CFG['main_song']] if stage=='main' else CFG['songs']):
  values=music_values(assets,song);N=assets.record(song,'B1')['frames'];music[str(song)]=assets.old['music'][str(song)]
  for name,v in values.items():arrays[f'music{song}_{name}']=v
  if stage=='controls' and song==CFG['main_song'] and not rebuild_main:
   path=ROOT/'main/graphs'/str(song);g,data=load(path)
   assert sha(path/'result.json')==read(ROOT/'main/result.json')['graphs'][str(song)]['result_sha256'];assert sha(path/'arrays.npz')==read(ROOT/'main/result.json')['graphs'][str(song)]['arrays_sha256']
  else:path=out/'graphs'/str(song);g,data=build(assets,song,path)
  graphs[str(song)]=dict(result_sha256=sha(path/'result.json'),arrays_sha256=sha(path/'arrays.npz'),geometry_trials=g['geometry_trials'],eligible_edges=g['eligible_edges'],candidate_counts=list(map(len,g['layers'])),path_capacity=path_capacity(g),fixed_path_capacity=path_capacity(g,True),fixed_intervals=g['base_intervals'])
  reward=values['boundary_reward'];rng=np.random.default_rng(CFG['seed']+song);shift=int(rng.integers(N//3,2*N//3));shuffled=np.roll(reward,shift);timing[str(song)]=dict(C3_reward_circular_shift=shift,same_graph=True,same_actual_energy=True,same_actual_E1W_beats=True,candidate_lattice_retains_frozen_B1_structure_prior=True,control_scope='INCREMENTAL_BOUNDARY_REWARD_CORRESPONDENCE_NOT_REMOVAL_OF_ALL_MUSIC')
  incumbent,_=choose(g,reward,assets.pop.reference,fixed=True);assert incumbent is not None
  for arm in (['B0','B1','C2'] if stage=='main' else ['B0','B1','C1','C2','C3']):
   if arm in ('B0','B1'):records.append(baseline(assets,song,arm,arrays));continue
   choice,search=choose(g,shuffled if arm=='C3' else reward,assets.pop.reference,fixed=arm=='C1',incumbent=incumbent if arm!='C1' else None);traj,record=assemble(assets,g,data,choice);record.update(song=song,arm=arm,frames=N,search=search)
   if traj is not None:
    for key,v in traj.items():arrays[f'song{song}_{arm}_{key}']=v
    record['metrics']=evaluate(assets.pop,traj,record,record['regions'],values,motion_fk);record['rotation_sha256']=array_hash(traj['R']);record['positions_sha256']=array_hash(traj['P']);record['execution_summary']=summarize_selected(record)
   records.append(record);print('R58_PLAN',song,arm,record['status'],flush=True)
 result=dict(uniform_profiles_sha256=sha(out/'uniform_profiles.json') if stage=='main' else read(ROOT/'main/result.json')['uniform_profiles_sha256'],stage=stage,records=records,graphs=graphs,timing_controls=timing,music=music,binding=dict(**binding(),parent=assets.pop.binding),neural_training_steps=0,optimizer_calls=0,world_root_contact='UNKNOWN',native24='NOT_EVALUABLE',can_route=False)
 if stage=='controls':
  old=read(ROOT/'main/result.json');z=np.load(ROOT/'main/arrays.npz')
  for r in old['records']:assert r==next(t for t in records if t['song']==r['song'] and t['arm']==r['arm'])
  for key in z.files:assert np.array_equal(arrays[key],z[key],equal_nan=True),key
 save_arrays(out/'arrays.npz',arrays);write(out/'result.json',result)
 if render:
  from .view import viewer
  viewer(out,assets.pop,arrays,result)
 return result

def support_audit(out):
 out=fresh(out);assets=Assets();old=read(PREVIOUS/'controls/result.json');rows=[]
 for r in old['factorial']:
  if r['status']=='SHARED_SUPPORT_FACTORIAL_DECOMPOSITION':continue
  song,k=r['song'],r['slot'];sources=[assets.record(song,a)['events'][k] for a in ('B0','B1')];cells=[]
  for i,e in enumerate(sources):
   unit=next(u for u in assets.bank if u['unit_uid']==e['event_uid']);lo,hi=unit['source_interval'];legal=[]
   for length in range(CFG['min_core'],CFG['max_core']+1):
    if .75-1e-10<=(length-1)/(hi-lo-1)<=1.33+1e-10:legal.append(length)
   for j,arm in enumerate(('B0','B1')):
    target=assets.record(song,arm)['events'][k]['target_interval'];ratio=(target[1]-target[0]-1)/(hi-lo-1);q=assets.node(unit,target,assets.pop.tracks[song]);cells.append(dict(source_factor=i,time_factor=j,event_uid=e['event_uid'],source_interval=[lo,hi],target_interval=target,global_target_source_ratio=ratio,legal=q is not None,reason=None if q is not None else 'OUTSIDE_ORIGINAL_GLOBAL_RATE_DOMAIN' if not .75-1e-10<=ratio<=1.33+1e-10 else 'ORIGINAL_MULTI_ANCHOR_INFEASIBLE',registered_alternative_lengths=legal))
  rows.append(dict(song=song,slot=k,cells=cells))
 assert len(rows)==14;write(out/'result.json',dict(records=rows,parent_sha256=sha(PREVIOUS/'controls/result.json'),can_route=False));return rows

def verify(out,source):
 out=fresh(out);source=Path(source);old=read(source/'result.json');current=binding();assert current['science']==old['binding']['science']
 from dunhuang.research.coverage_r34.common import forbid_optimizers
 forbid_optimizers();new=run(out/'rebuild',old['stage'],False,True);new['binding']['commit']=old['binding']['commit'];assert new==old;assert sha(source/'arrays.npz')==sha(out/'rebuild/arrays.npz')
 for name,digest in current['dependencies'].items():assert __import__('hashlib').sha256(subprocess.check_output(['git','show',old['binding']['commit']+':'+name])).hexdigest()==digest
 for name,digest in current['science'].items():assert __import__('hashlib').sha256(subprocess.check_output(['git','show',old['binding']['commit']+':src/dunhuang/research/execution_planning_r58/'+name])).hexdigest()==digest
 write(out/'result.json',dict(result_exact=True,arrays_bytes_exact=True,all_graphs_rebuilt_from_original_sources=True,source_commit=old['binding']['commit'],replay_commit=current['commit'],optimizer_calls=0,model_calls=0,can_route=False))
