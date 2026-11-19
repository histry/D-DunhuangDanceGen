from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
from .common import *
from dunhuang.research.geometry_recovery_r57.study import bridge

def boundary_lattice(base):
 frames=base[-1]['interval'][1];ends=[r['interval'][1] for r in base if r['kind']=='core'];options=[]
 for i,end in enumerate(ends):options.append(sorted(set(end+d for d in (CFG['boundary_offsets'] if i in CFG['variable_boundary_indices'] else [0]) if 0<end+d<=frames)))
 options[-1]=[frames];intervals=[]
 for k,values in enumerate(options):
  starts=[0] if k==0 else [e+CFG['gap'] for e in options[k-1]];intervals.append(sorted([a,b] for a in starts for b in values if CFG['min_core']<=b-a<=CFG['max_core']))
 return intervals
def edge_job(args):
 i,j,L,R,J,parents,reference=args;curve,e=bridge(L,R,CFG['gap'],SimpleNamespace(joints=J,parents=parents),reference);return i,j,curve,e
def build(assets,song,out):
 out=fresh(out);base=assets.record(song,'B1')['regions'];track=assets.pop.tracks[song];intervals=boundary_lattice(base);layers=[];arrays={};reject=[]
 for slot,spans in enumerate(intervals):
  layer=[]
  for target in spans:
   for u in assets.bank:
    result=assets.node(u,target,track)
    if result is None:reject.append(dict(slot=slot,target=target,event_uid=u['unit_uid'],reason='ORIGINAL_E1W_RATE_OR_PHASE_INFEASIBLE'));continue
    n,R=result;layer.append(n);arrays[n['id']]=R
  layer.sort(key=lambda n:(n['target'],-n['heuristic'],n['event_uid']));layers.append(layer)
 jobs=[]
 for left,right in zip(layers,layers[1:]):
  for a in left:
   for b in right:
    if a['target'][1]+CFG['gap']==b['target'][0] and a['source_uid']!=b['source_uid']:jobs.append((a['id'],b['id'],arrays[a['id']],arrays[b['id']],assets.pop.net.joints,assets.pop.net.parents,assets.pop.reference))
 if len(jobs)>CFG['maximum_geometry_trials_per_song']:raise ValueError('Registered geometry budget exceeded')
 edges=[]
 with ProcessPoolExecutor(max_workers=CFG['workers']) as pool:
  for i,j,Q,e in pool.map(edge_job,jobs,chunksize=8):
   key=identity([i,j])[:24];edges.append(dict(left=i,right=j,key=key,eligible=Q is not None,evidence=e))
   if Q is not None:arrays['edge_'+key]=Q
   if len(edges)%500==0:print('R58_EDGES',song,len(edges),len(jobs),flush=True)
 result=dict(song=song,layers=layers,edges=edges,rejections=reject,geometry_trials=len(jobs),eligible_edges=sum(e['eligible'] for e in edges),intervals=intervals,base_intervals=[r['interval'] for r in base if r['kind']=='core']);save_arrays(out/'arrays.npz',arrays);write(out/'result.json',result);return result,arrays
def load(path):
 path=Path(path);return read(path/'result.json'),np.load(path/'arrays.npz')

def choose(graph,reward,reference,fixed=False,execution=True,structure=True,incumbent=None):
 edges={(e['left'],e['right']):e for e in graph['edges'] if e['eligible']};lookup={n['id']:n for layer in graph['layers'] for n in layer};total=graph['layers'][-1][0]['target'][1]-CFG['gap']*(CFG['events']-1);target=total/CFG['events'];states=[dict(score=0.,ids=[],units={},sources={})];counts=[]
 def extend(state,node):
  uid,src=node['event_uid'],node['source_uid'];a,b=node['target'];value=node['heuristic']-CFG['unit_reuse']*state['units'].get(uid,0)-CFG['source_reuse']*state['sources'].get(src,0)
  if execution:value-=CFG['execution_weight']*node['execution_cost']
  if structure:
   value-=CFG['length_weight']*((b-a-target)/(CFG['max_core']-CFG['min_core']))**2
   if len(state['ids'])<CFG['events']-1:value+=CFG['structure_weight']*float(reward[b+CFG['gap']//2])
  if state['ids']:
   e=edges.get((state['ids'][-1],node['id']))
   if e is None:return None
   m=e['evidence'];value-=CFG['edge_weight']*m['methods'][m['selected']]['gap_D3']/max(reference['D3'],1e-9)
  units=dict(state['units']);units[uid]=units.get(uid,0)+1;sources=dict(state['sources']);sources[src]=sources.get(src,0)+1
  return dict(score=state['score']+value,ids=state['ids']+[node['id']],units=units,sources=sources)
 for k,layer in enumerate(graph['layers']):
  options=[n for n in layer if not fixed or n['target']==graph['base_intervals'][k]];expanded=[]
  for state in states:
   for n in options:
    q=extend(state,n)
    if q is not None:expanded.append(q)
  expanded.sort(key=lambda s:(-s['score'],s['ids']));states=expanded[:CFG['beam']];counts.append(dict(layer=k,expanded=len(expanded),retained=len(states)))
  if not states:break
 result=states[0] if states and len(states[0]['ids'])==CFG['events'] else None;protected=False
 if incumbent is not None:
  state=dict(score=0.,ids=[],units={},sources={})
  for key in incumbent['ids']:
   state=extend(state,lookup[key]);assert state is not None
  if result is None or state['score']>result['score']:result=state;protected=True
 return result,dict(layers=counts,protected_C1_incumbent_selected=protected,global_optimality_claim=False,beam=CFG['beam'])
def path_capacity(graph,fixed=False):
 edges={(r['left'],r['right']) for r in graph['edges'] if r['eligible']};states={}
 for k,layer in enumerate(graph['layers']):
  options=[n for n in layer if not fixed or n['target']==graph['base_intervals'][k]];states={n['id']:(1 if not k else sum(v for uid,v in states.items() if (uid,n['id']) in edges)) for n in options}
 return str(sum(states.values()))
