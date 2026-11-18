"""Staged actual-source experiments; no training and no geometry threshold changes."""
import copy,time
try:import resource
except ImportError:resource=None
from .common import *
from .population import Population
from .bridge_graph import Graph,calculate,adjacency
from .coupled_search import solve,layers_for,compose,evaluate
from dunhuang.research.complete_r36.core import multi_warp

def baseline(pop):
 counts={}
 for root in (R51,R52):
  files=read(root/('publication/seal.json'))['files']
  for f in files:assert Path(f['path']).stat().st_size==f['bytes'] and sha(f['path'])==f['sha256'],f['path']
  counts[str(root)]=len(files)
 records=[];frozen={r['key']:r for r in pop.r52['records']}
 with np.load(R52/'plans/arrays.npz') as z:
  for row in pop.old['records']:
   pop._music_sha=pop.tracks[row['song']]['audio_sha256'];pop._structure_sha=pop.tracks[row['song']]['structure_sha256'];pop._beats=pop.tracks[row['song']]['structure']['beats'];parts=[];ids=[];chosen=row['selected']['HeuristicFullPool'];cores=[s['interval'] for s in row['regions'] if s['kind']=='core']
   assert chosen['events']==next(iter(frozen[row['key']]['selected'].values()))['events']
   for slot,e in enumerate(chosen['events']):
    c=next(c for c in row['pools'][slot] if c['unit']['unit_uid']==e['event_uid']);a,b=cores[slot];pop._activity=float(np.mean(pop.tracks[row['song']]['structure']['energy_percentile'][a:b]));phases=[dict(p,type=p['body_region']+'_'+p['type']) for p in c['unit']['phases']];assert multi_warp(e['source_interval'],[a,b],phases,pop.tracks[row['song']]['structure']['beats'])==e['warp'];ids.append(pop.node(c,[a,b],e['warp']))
   for slot,key in enumerate(ids):
    parts.append(pop.R[key])
    if slot<len(ids)-1:
     curve,e=calculate((pop.R[key][-4:],pop.R[ids[slot+1]][:4],row['gap'],pop.net.joints,pop.net.parents,pop.reference));assert curve is not None;parts.append(curve);old=copy.deepcopy(chosen['connections'][slot])
     for m in old['methods'].values():m.pop('jerk_curve',None)
     assert {k:v for k,v in e.items() if k!='failure_reasons'}==old
   R=np.concatenate(parts);prefix=row['key']+'_Heuristic_repeat0_same0_balanced0';assert np.array_equal(R,z[prefix+'_R']) and np.array_equal(relative_fk(R,pop.net.joints,pop.net.parents).astype('<f4'),z[prefix]);records.append(dict(key=row['key'],events=chosen['events'],rotation_sha256=array_hash(R),unchanged=True))
 return dict(records=records,sealed_files_verified=counts,neural_training_steps=0,can_route=False)

def jet_delta(pop,left,right,base_left,base_right):
 l=np.asarray(pop.nodes[left]['endpoint_jets'])[1]-np.asarray(pop.nodes[base_left]['endpoint_jets'])[1];r=np.asarray(pop.nodes[right]['endpoint_jets'])[0]-np.asarray(pop.nodes[base_right]['endpoint_jets'])[0];d=np.stack((l,r));return dict(omega_rms=float(np.sqrt(np.mean(d[:,0]**2))),alpha_rms=float(np.sqrt(np.mean(d[:,1]**2))))

def mechanism(pop,graph,lefts,rights,rows):
 lookup={(r['left'],r['right']):r['bridge_id'] for r in rows};base=graph.records[lookup[lefts[0],rights[0]]];cells=[]
 for i,l in enumerate(lefts):
  for j,r in enumerate(rights):
   key=lookup[l,r];e=graph.records[key];delta=jet_delta(pop,l,r,lefts[0],rights[0]);admitted=e['selected'] is not None;base_ok=base['selected'] is not None;flip=admitted!=base_ok;quality=False;numeric=False
   if admitted and base_ok:quality=significant(base['methods'][base['selected']]['gap_D3'],e['methods'][e['selected']]['gap_D3'])
   for method in set(base['methods']) & set(e['methods']):
    numeric |= any(abs(base['methods'][method][k]-e['methods'][method][k])>1e-8 for k in ('gap_D2','gap_D3','boundary_D3'))
   cells.append(dict(left=i,right=j,intervention='BASELINE' if i==j==0 else 'LEFT_ONLY' if j==0 else 'RIGHT_ONLY' if i==0 else 'JOINT',bridge_id=key,admitted=admitted,G0_chart=e['G0']['chart_eligible'],G1_chart=e['G1']['status'].startswith('GEOMETRICALLY_ADMITTED'),jet_delta=delta,numeric_effect=bool(max(delta.values())>CFG['numerical_jet_tolerance'] and (numeric or flip or e['G0']['chart_eligible']!=base['G0']['chart_eligible'])),admission_flip=flip,meaningful_quality_change=quality,mechanism_effect=bool(max(delta.values())>CFG['numerical_jet_tolerance'] and (flip or quality))))
 default=max(range(len(cells)),key=lambda k:(cells[k]['jet_delta']['alpha_rms'],cells[k]['jet_delta']['omega_rms'],-k))
 return dict(cells=cells,default_cell=default,has_effect=any(c['mechanism_effect'] for c in cells),has_numeric_effect=any(c['numeric_effect'] for c in cells))

def run(out,mode):
 out=fresh(out);started=time.perf_counter();pop=Population();base=baseline(pop);write(out/'baseline.json',base);study_root=ROOT/'replay' if out.parent==ROOT/'replay' else ROOT;parent=study_root/'preview' if mode=='main' else study_root/'main' if mode=='controls' else None
 if mode!='preview':assert read(ROOT/'receipts/preview_browser.json')['passed']
 if mode=='controls':assert read(ROOT/'receipts/main_browser.json')['passed'] and read(ROOT/'main/result.json')['gates']['P2_investment_gate']
 graph=Graph(pop,parent=parent);records=[];pairs=[];arrays=dict(J=pop.net.joints,parents=pop.net.parents);case_resources=[];search_resources=[];negative_calls=0
 try:
  for row in pop.old['records']:
   t=time.perf_counter();banks=pop.banks(row,controls=mode=='controls');old=row['selected']['HeuristicFullPool'];case=dict(key=row['key'],song=row['song'],gap=row['gap'],regions=row['regions'],music_interval=row['music_interval'],banks=banks,plans={})
   if mode=='preview':
    for slot in range(len(banks)-1):
     l=next(e for e in banks[slot] if e['event_uid']==old['events'][slot]['event_uid']);r=next(e for e in banks[slot+1] if e['event_uid']==old['events'][slot+1]['event_uid']);lefts=l['sets']['top'];rights=r['sets']['top'];erows=graph.add([(a,b) for a in lefts for b in rights],row['gap']);m=mechanism(pop,graph,lefts,rights,erows);neg=[]
     for side,event in [('left',l),('right',r)]:
      if not event['negative_control']:neg.append(dict(side=side,status='NOT_CONSTRUCTIBLE_IN_ORIGINAL_LATTICE'));continue
      a=event['negative_control'] if side=='left' else lefts[0];b=event['negative_control'] if side=='right' else rights[0];curve,e=calculate((pop.R[a][-4:],pop.R[b][:4],row['gap'],pop.net.joints,pop.net.parents,pop.reference));negative_calls+=1;zero=graph.records[m['cells'][0]['bridge_id']];assert e==zero and np.array_equal(curve,graph.curves[m['cells'][0]['bridge_id']]);neg.append(dict(side=side,status='DIRECT_RECONSTRUCTION_EXACT_BRIDGE_AND_SCREEN',left=a,right=b))
     cores=[s['interval'] for s in row['regions'] if s['kind']=='core'];pairs.append(dict(key=row['key']+'_edge'+str(slot),song=row['song'],gap=row['gap'],music_interval=[cores[slot][0],cores[slot+1][1]],lefts=lefts,rights=rights,negative_controls=neg,**m))
   else:
    names=('top','random','linear') if mode=='controls' else ('top',);layers,edges,edge_rows=adjacency(pop,graph,banks,row['gap'],names);case['edges']=edge_rows
    # Include the baseline fixed event-pair observations in every complete candidate graph.
    for slot in range(len(banks)-1):
     le=next(e for e in banks[slot] if e['event_uid']==old['events'][slot]['event_uid']);re=next(e for e in banks[slot+1] if e['event_uid']==old['events'][slot+1]['event_uid']);ls=le['sets']['top'];rs=re['sets']['top'];rr=[e for e in edge_rows if e['slot']==slot and e['left'] in ls and e['right'] in rs];pairs.append(dict(key=row['key']+'_edge'+str(slot),**mechanism(pop,graph,ls,rs,rr)))
   records.append(case);case_resources.append(dict(key=row['key'],seconds=time.perf_counter()-t));print('R53',mode,row['key'],'canonical',len(graph.records),'calls',graph.calls,flush=True)
  # Full P1 population, including rejected W0 edges, not only the displayed Heuristic pairs.
  full_effects=[]
  if mode!='preview':
   for case in records:
    for slot in range(len(case['banks'])-1):
     emap={(e['left'],e['right']):e['bridge_id'] for e in case['edges'] if e['slot']==slot}
     for l in case['banks'][slot]:
      for r in case['banks'][slot+1]:
       if l['source_uid']==r['source_uid']:continue
       ls=l['sets']['top'];rs=r['sets']['top'];rr=[dict(left=a,right=b,bridge_id=emap[a,b]) for a in ls for b in rs];m=mechanism(pop,graph,ls,rs,rr);full_effects.append(dict(case=case['key'],slot=slot,left_event=l['event_uid'],right_event=r['event_uid'],**m))
  G1=any(e['max_sampled_motion_delta']>CFG['numerical_motion_tolerance'] for c in records for layer in c['banks'] for e in layer);G2=G1 and any(p['has_numeric_effect'] for p in (full_effects or pairs));investment=G1 and any(p['has_effect'] for p in (full_effects or pairs))
  if mode!='preview':
   for case,row in zip(records,pop.old['records']):
    banks=case['banks'];edges={(e['slot'],e['left'],e['right']):e['bridge_id'] for e in case['edges']}
    arms={'B0':('fixed',1),'B3':('top',4)}
    if mode=='controls':arms.update(B1=('independent',4),B2=('random',4),W3=('linear',1),B3_K1=('top',1),B3_K2=('top',2),B2_K1=('random',1),B2_K2=('random',2))
    gate=investment
    for arm,(strategy,K) in arms.items():
     if arm!='B0' and not gate:case['plans'][arm]=dict(status='NOT_RUN_G2_FAILED');continue
     search_start=time.perf_counter();chosen_layers=layers_for(pop,banks,strategy,K);solution=solve(pop,graph,chosen_layers,edges);solution['candidate_counts']=list(map(len,chosen_layers));required=[edges[i,l,r] for i,(left,right) in enumerate(zip(chosen_layers,chosen_layers[1:])) for l in left for r in right if (i,l,r) in edges];solution['required_edges']=len(required);solution['required_unique_bridge_readsets']=len(set(required));search_resources.append(dict(case=row['key'],arm=arm,seconds=time.perf_counter()-search_start));R=compose(pop,graph,solution)
     if R is not None:
      solution['metrics']=evaluate(R,row,pop);arrays[row['key']+'_'+arm+'_R']=R;arrays[row['key']+'_'+arm]=relative_fk(R,pop.net.joints,pop.net.parents).astype('<f4')
      if arm in ('B0','B3_K1','B2_K1'):
       with np.load(R52/'plans/arrays.npz') as z:assert np.array_equal(R,z[row['key']+'_Heuristic_repeat0_same0_balanced0_R'])
     case['plans'][arm]=solution
  graph.save(out);save_arrays(out/'arrays.npz',arrays)
  write(out/'result.json',dict(mode=mode,binding=pop.binding,baseline='baseline.json',records=records,pairs=pairs,full_pair_mechanisms=full_effects,nodes=pop.nodes,gates=dict(G1=G1,G2=G2,P2_investment_gate=investment,G3='PENDING_EQUAL_BUDGET_CONTROLS',G4='NOT_TESTED_EXPOSED_DEVELOPMENT_MUSIC_ONLY'),negative_control_direct_calls=negative_calls,neural_training_steps=0,whole_song_generations=0,root_contact='UNKNOWN',can_route=False))
  graph.close()
  write(out/'resources.json',dict(seconds=time.perf_counter()-started,parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None,child_max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss if resource else None,workers=CFG['workers'],bridge_new_evaluations=graph.calls,exact_readset_hits=graph.hits,cases=case_resources,searches=search_resources,scope='TIMING_NOT_PART_OF_EXACT_SCIENTIFIC_COMPARISON'))
  if mode=='preview':
   from .viewer import preview
   preview(out,pop,graph,pairs)
  else:
   from .viewer import plans
   plans(out,pop)
 finally:graph.close()
