"""Full warp keys plus certified exact four-frame read-set reuse, never UID caching."""
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import numpy as np
from .common import *
from dunhuang.research.transition_planning_r50.planning import bridge

def calculate(args):
 left,right,gap,J,parents,reference=args
 curve,evidence=bridge(left,right,gap,SimpleNamespace(joints=J,parents=parents),reference)
 for value in evidence['methods'].values():value.pop('jerk_curve',None)
 evidence['failure_reasons']={name:([reason for reason,bad in [('D2_REFERENCE',m['gap_D2']>reference['D2']),('D3_REFERENCE',m['gap_D3']>reference['D3']),('SHOULDER_WRIST_REFERENCE',m['shoulder_wrist_min_m']<reference['shoulder_wrist_lower_m'])] if bad]) for name,m in evidence['methods'].items()}
 if 'G0' not in evidence['methods']:evidence['failure_reasons']['G0']=['CHART_REJECTED']
 if 'G1' not in evidence['methods']:evidence['failure_reasons']['G1']=['REGISTERED_SEGMENTS_REJECTED']
 return curve,evidence

class Graph:
 def __init__(self,pop,parent=None,workers=None):
  self.pop=pop;self.records={};self.curves={};self.full={};self.calls=0;self.hits=0;self.parent_ids=set();self.executor=ProcessPoolExecutor(max_workers=workers or CFG['workers']);self.policy=identity(dict(schema='R53_EXACT_READSET_V1',protocol=pop.binding['protocol_sha256'],algorithms=pop.binding['algorithms'],reference=pop.reference,J=pop.binding['J_sha256'],parents=pop.binding['parents_sha256']))
  if parent:
   old=read(parent/'cache.json');assert old['policy']==self.policy
   assert sha(parent/'cache_arrays.npz')==old['arrays_sha256']
   self.records=old['records'];self.full=old['full_keys'];self.parent_ids=set(self.records)
   with np.load(parent/'cache_arrays.npz') as z:self.curves={k:z[k] for k in z.files}
 def add(self,pairs,gap):
  pending={};rows=[]
  for left,right in pairs:
   l,r=self.pop.nodes[left],self.pop.nodes[right];key=identity(dict(left=l['id'],right=r['id'],gap=gap,policy=self.policy));boundary=np.concatenate((self.pop.R[left][-4:],self.pop.R[right][:4]));readset=identity(dict(boundary=array_hash(boundary),gap=gap,policy=self.policy));self.full[key]=readset;rows.append(dict(id=key,left=left,right=right,bridge_id=readset))
   if readset not in self.records and readset not in pending:pending[readset]=(boundary[:4],boundary[4:],gap,self.pop.net.joints,self.pop.net.parents,self.pop.reference)
   else:self.hits+=1
  for key,(curve,evidence) in zip(pending,self.executor.map(calculate,pending.values())):
   self.records[key]=evidence
   if curve is not None:self.curves[key]=curve
  self.calls+=len(pending);return rows
 def save(self,out):
  save_arrays(out/'cache_arrays.npz',self.curves);write(out/'cache.json',dict(policy=self.policy,records=self.records,full_keys=self.full,arrays_sha256=sha(out/'cache_arrays.npz'),readset_scope='EXACT_ALL_FOUR_POSE_SAMPLES_PER_SIDE_PLUS_J_PARENTS_REFERENCE_ALGORITHM_GAP',counts=dict(new_bridge_constructions=self.calls,readset_hits=self.hits,parent_records=len(self.parent_ids))))
 def close(self):
  if self.executor is not None:self.executor.shutdown();self.executor=None

def adjacency(pop,graph,banks,gap,names=('top',)):
 layers=[]
 for layer in banks:layers.append(list(dict.fromkeys(k for e in layer for name in names for k in e['sets'][name])))
 edges={};rows=[]
 for slot,(left,right) in enumerate(zip(layers,layers[1:])):
  pairs=[(l,r) for l in left for r in right if pop.nodes[l]['source_uid']!=pop.nodes[r]['source_uid']]
  for row in graph.add(pairs,gap):edges[slot,row['left'],row['right']]=row['bridge_id'];rows.append(dict(slot=slot,**row))
 return layers,edges,rows
