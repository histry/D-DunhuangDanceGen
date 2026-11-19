"""Whole-song old/recovered graph comparison on exactly the frozen R56 pools."""
from .common import *
from .profiles import Assets
from dunhuang.research.music_structure_r56.planning import choose as frozen_choose
from dunhuang.research.music_structure_r56.common import kernel
from dunhuang.research.music_structure_r56.music import descriptors
from dunhuang.research.geometry_recovery_r57.study import bridge
from dunhuang.research.validation_r51.common import regions
def capacity(pools,edges):
 values=[1]*len(pools[0])
 for k,pool in enumerate(pools[1:]):values=[sum(v for i,v in enumerate(values) if edges.get((k,i,j),(None,))[0] is not None) for j in range(len(pool))]
 return str(sum(values))
def run(out):
 out=fresh(out);a=Assets();repairs=read(PREVIOUS/'repair/result.json')['records'];recovered={(r['graph_key'],r['slot'],r['left'],r['right']):r['evidence'] for r in repairs};rows=[];arrays=dict(J=a.pop.net.joints,parents=a.pop.net.parents)
 for record in a.old['records']:
  song,arm=record['song'],record['arm'];gkey=record['graph_key'];g=a.old['graphs'][gkey];pools=g['pools'];old={(e['slot'],e['left'],e['right']):(True if e['selected'] is not None else None,e) for e in g['edges']};new=dict(old);added=[]
  for e in g['edges']:
   key=gkey,e['slot'],e['left'],e['right']
   if key in recovered:new[e['slot'],e['left'],e['right']]=(True,recovered[key]);added.append([e['slot'],e['left'],e['right']])
  sim=kernel(descriptors(a.z[f'music{song}_chroma'],record['regions'])) if arm=='B3' else None;motion=a.z['motion_objective_similarity'] if arm=='B3' else None
  prior,_=frozen_choose(pools,old,a.pop.reference,sim,motion);assert [pools[k][i]['unit']['unit_uid'] for k,i in enumerate(prior['ids'])]==[e['event_uid'] for e in record['events']];assert prior['score']==record['score'];choice,search=frozen_choose(pools,new,a.pop.reference,sim,motion);same=choice['ids']==prior['ids'];chunks=[];spans=[r['interval'] for r in record['regions'] if r['kind']=='core'];events=[]
  for k,i in enumerate(choice['ids']):
   n=pools[k][i];w=np.asarray(n['warp']['time_warp']);target=spans[k];q=np.interp(np.arange(*target),w[:,1],w[:,0]);R=rotation_sample(a.pop.lib.rotations[n['unit']['source_uid']],q);chunks.append(R);events.append(n['unit']['unit_uid'])
   if k<len(spans)-1:
    right=pools[k+1][choice['ids'][k+1]];v=np.asarray(right['warp']['time_warp']);Q=rotation_sample(a.pop.lib.rotations[right['unit']['source_uid']],np.interp(np.arange(*spans[k+1]),v[:,1],v[:,0]));curve,e=bridge(R,Q,CFG['gap'],a.pop.net,a.pop.reference);assert curve is not None;chunks.append(curve)
  R=np.concatenate(chunks);P=relative_fk(R,a.pop.net.joints,a.pop.net.parents).astype('<f4');key=f'song{song}_{arm}';arrays[key+'_R']=R;arrays[key+'_P']=P
  if same:assert np.array_equal(R,a.z[key+'_R']) and np.array_equal(P,a.z[key+'_P'])
  rows.append(dict(song=song,arm=arm,old_eligible=g['eligible'],new_eligible=g['eligible']+len(added),added_edges=added,old_path_capacity=capacity(pools,old),new_path_capacity=capacity(pools,new),selected_path_changed=not same,old_events=[e['event_uid'] for e in record['events']],new_events=events,old_score=prior['score'],new_score=choice['score'],original_unchanged=True,kinematics=regions(R,a.pop.net.joints,a.pop.net.parents,record['regions'])))
  print('R58_G1_GRAPH',song,arm,len(added),not same,flush=True)
 result=dict(records=rows,binding=binding(),old_result_sha256=sha(PARENT/'controls/result.json'),repair_sha256=sha(PREVIOUS/'repair/result.json'),new_neural_samples=0,can_route=False,physical_qualified=False);write(out/'result.json',result);save_arrays(out/'arrays.npz',arrays);return result
def verify(out,source):
 out=fresh(out);old=read(Path(source)/'result.json');new=run(out/'rebuild');new['binding']['commit']=old['binding']['commit'];assert new==old;assert sha(Path(source)/'arrays.npz')==sha(out/'rebuild/arrays.npz');write(out/'result.json',dict(result_exact=True,arrays_bytes_exact=True,optimizer_calls=0))
