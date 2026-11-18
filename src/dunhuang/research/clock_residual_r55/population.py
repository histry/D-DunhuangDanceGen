"""Remove held calibration identities from BOTH training candidate arms."""
from copy import deepcopy
import hashlib
import numpy as np

def independent_candidates(meta,split,seed):
 tasks=deepcopy(meta['tasks']);rows=meta['rows'];cal=set(split['calibration']);opt=set(split['optimization']);audit=dict(optimization_tasks=0,training_admitted=0,excluded_no_negatives=[],candidate_removals=0,matched_cardinality=True,held_training_identity_intersection=[])
 held={key:{r.get(key,r['source_group']) for r in rows if r['source_group'] in cal} for key in ('raw_motion_sha256','audio_sha256')}
 def protected(row):return row['source_group'] in cal or any(row.get(key,row['source_group']) in values for key,values in held.items())
 for t in tasks:
  t['training_admitted']=False
  if t['role']!='train' or t['group'] not in opt:continue
  audit['optimization_tasks']+=1;parts={}
  if protected(rows[t['query']]):audit['excluded_no_negatives'].append(t['uid']);continue
  for prefix in ('','unconditional_'):
   items=list(zip(t[prefix+'candidates'],t[prefix+'positive'],t[prefix+'heuristic']));kept=[x for x in items if not protected(rows[x[0]])];audit['candidate_removals']+=len(items)-len(kept);parts[prefix]=kept
  if any(not any(p for _,p,_ in v) or not any(not p for _,p,_ in v) for v in parts.values()):audit['excluded_no_negatives'].append(t['uid']);continue
  assert sorted(j for j,p,_ in parts[''] if p)==sorted(j for j,p,_ in parts['unconditional_'] if p)
  negative_count=min(sum(not p for _,p,_ in v) for v in parts.values())
  for prefix,items in parts.items():
   negatives={j for j,p,_ in [x for x in items if not x[1]][:negative_count]};items=[x for x in items if x[1] or x[0] in negatives];t[prefix+'candidates']=[j for j,_,_ in items];t[prefix+'positive']=[bool(p) for _,p,_ in items];t[prefix+'heuristic']=[h for _,_,h in items]
  assert len(t['candidates'])==len(t['unconditional_candidates'])
  labels=np.array(t['positive'],bool);rng=np.random.default_rng(int(hashlib.sha256((str(seed)+t['uid']).encode()).hexdigest()[:16],16));shuffled=labels[rng.permutation(len(labels))]
  if np.array_equal(shuffled,labels):shuffled=np.roll(labels,1)
  assert not np.array_equal(shuffled,labels) and shuffled.sum()==labels.sum();t['shuffled_positive']=shuffled.tolist();t['training_admitted']=True;audit['training_admitted']+=1
 used={rows[j]['source_group'] for t in tasks if t['training_admitted'] for j in [t['query']]+t['candidates']+t['unconditional_candidates']};audit['held_training_identity_intersection']=sorted(used&cal);audit['active_optimization_groups']=sorted({t['group'] for t in tasks if t['training_admitted']});assert not used&cal
 for key,values in held.items():
  overlap={rows[j].get(key,rows[j]['source_group']) for t in tasks if t['training_admitted'] for j in [t['query']]+t['candidates']+t['unconditional_candidates']}&values;assert not overlap;audit['held_'+key+'_intersection']=sorted(overlap)
 return tasks,audit
