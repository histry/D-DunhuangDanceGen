"""Audit frozen R58 score terms and missing execution evidence, without changing paths or weights."""
from .common import *
from dunhuang.research.execution_planning_r58.common import CFG as OLD

def evidence_cost(ratios):
 values=[abs(float(np.log(max(v['ratio'],1e-8))))/int(k) for k,v in ratios.items() if v['ratio'] is not None]
 return dict(status='OBSERVED' if values else 'UNKNOWN',value=float(np.mean(values)) if values else None,observed_orders=len(values))

def decomposition(nodes,connections,reward,reference):
 terms=dict(heuristic=0.,execution=0.,length=0.,boundary=0.,geometry=0.,unit_reuse=0.,source_reuse=0.);units={};sources={};target=(nodes[-1]['target'][1]-OLD['gap']*(len(nodes)-1))/len(nodes);rows=[]
 for k,n in enumerate(nodes):
  uid,src=n['event_uid'],n['source_uid'];a,b=n['target'];v=dict(heuristic=n['heuristic'],execution=-OLD['execution_weight']*n['execution_cost'],length=-OLD['length_weight']*((b-a-target)/(OLD['max_core']-OLD['min_core']))**2,boundary=OLD['structure_weight']*float(reward[b+OLD['gap']//2]) if k<len(nodes)-1 else 0.,geometry=0.,unit_reuse=-OLD['unit_reuse']*units.get(uid,0),source_reuse=-OLD['source_reuse']*sources.get(src,0))
  if k:
   e=connections[k-1];v['geometry']=-OLD['edge_weight']*e['methods'][e['selected']]['gap_D3']/max(reference['D3'],1e-9)
  for name,x in v.items():terms[name]+=float(x)
  units[uid]=units.get(uid,0)+1;sources[src]=sources.get(src,0)+1;rows.append(dict(slot=k,event_uid=uid,target=n['target'],terms=v,evidence=evidence_cost(n['source_normalized'])))
 return dict(terms=terms,total=sum(terms.values()),slots=rows)

def run(out):
 out=fresh(out);root=Path('outputs/execution_planning_r58_01');seal=read(root/'artifacts_manifest.json')['files'];needed=['controls/result.json','controls/arrays.npz','main/uniform_profiles.json']+[(('main' if s==0 else 'controls')+'/graphs/'+str(s)+'/result.json') for s in range(3)]
 for name in needed:assert sha(root/name)==next(r['sha256'] for r in seal if r['path']==name)
 data=read(root/'controls/result.json');z=np.load(root/'controls/arrays.npz')
 # The frozen graph uses the same R53 population reference, not an estimated replacement.
 from dunhuang.research.warp_coupling_r53.population import Population
 reference=Population().reference;records=[];missing=[];selected_missing=[]
 for song in range(3):
  g=read(root/('main' if song==0 else 'controls')/'graphs'/str(song)/'result.json');lookup={(n['event_uid'],tuple(n['target'])):n for layer in g['layers'] for n in layer}
  for n in lookup.values():
   evidence=evidence_cost(n['source_normalized'])
   if evidence['status']=='UNKNOWN':missing.append(dict(song=song,node=n['id'],event_uid=n['event_uid'],execution_cost=n['execution_cost']))
  for r in [r for r in data['records'] if r['song']==song and r['arm'] in ('B1','C1','C2','C3')]:
   nodes=[lookup[e['event_uid'],tuple(e['target_interval'])] for e in r['events']];reward=z[f'music{song}_boundary_reward'];shift=data['timing_controls'][str(song)]['C3_reward_circular_shift'];reward=np.roll(reward,shift) if r['arm']=='C3' else reward;result=decomposition(nodes,r['connections'],reward,reference)
   if r['arm']!='B1':assert abs(result['total']-r['score'])<1e-8,(song,r['arm'],result['total'],r['score'])
   records.append(dict(song=song,arm=r['arm'],saved_score=r['score'],common_objective=result,original_score_reconstructed=r['arm']!='B1',scope='B1_COMMON_RESCORE_NOT_REWRITING_OLD_OBJECTIVE'))
   selected_missing.extend(dict(song=song,arm=r['arm'],slot=k) for k,n in enumerate(nodes) if evidence_cost(n['source_normalized'])['status']=='UNKNOWN')
 uniform=read(root/'main/uniform_profiles.json');uniform_missing=[dict(event_uid=r['event_uid'],duration=r['duration']) for r in uniform if r['legal'] and evidence_cost(r['source_normalized'])['status']=='UNKNOWN'];changes=[]
 for song in range(3):
  a=next(r for r in records if r['song']==song and r['arm']=='C1');b=next(r for r in records if r['song']==song and r['arm']=='C2');changes.append(dict(song=song,terms={k:b['common_objective']['terms'][k]-a['common_objective']['terms'][k] for k in a['common_objective']['terms']},total=b['common_objective']['total']-a['common_objective']['total']))
 result=dict(binding=binding(),parent_manifest_sha256=sha(root/'artifacts_manifest.json'),records=records,C2_minus_C1=changes,all_unknown_graph_nodes=missing,all_unknown_selected_nodes=selected_missing,all_unknown_uniform_entries=uniform_missing,missing_is_not_low_cost=True,frozen_R58_fallback_zero_retained=True,production_changed=False,optimizer_calls=0);write(out/'result.json',result);print('R59_R58_DECOMPOSED',len(records),len(missing),len(selected_missing),flush=True);return result
