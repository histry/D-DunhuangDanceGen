from itertools import product
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import multi_warp,rotation_sample
from dunhuang.research.warp_coupling_r53.warp_lattice import lattice,best,suffix,unrank,rank_of,variants,validate,legal
from dunhuang.research.warp_coupling_r53.bridge_graph import calculate
from dunhuang.research.warp_coupling_r53.coupled_search import solve
from dunhuang.research.warp_coupling_r53.common import save_arrays

def graph():
 phases=[dict(type='upper_speed_peak',frame=x) for x in (30,60,90,120)]
 return lattice([0,161],[500,661],phases,list(range(500,661,6))),phases

def test_kbest_is_original_dp_and_matches_complete_small_lattice():
 g,phases=graph();brute=[]
 for ids in product(*[range(len(v)) for v in g['options']]):
  if not all(legal(g,k,ids[k-1],ids[k]) for k in range(1,len(ids))):continue
  w=[[x,g['options'][k][j]] for k,(x,j) in enumerate(zip(g['source'],ids))];cost=0.
  for k,j in enumerate(ids[1:],1):cost=cost+g['costs'][k][j][0]+g['costs'][k][j][1]
  brute.append((cost,w))
 brute.sort();top=best(g,4);assert [(x['cost'],x['time_warp']) for x in top]==brute[:4];assert top[0]==multi_warp([0,161],[500,661],phases,list(range(500,661,6)));counts=suffix(g);assert counts[0][0]==len(brute)
 for rank in range(counts[0][0]):assert rank_of(g,unrank(g,rank),counts)==rank
 a=variants(g,4,53);assert a==variants(g,4,53) and len(a['random'])==len(a['top']);assert a['random'][0]==a['top'][0]
 for w in a['random']+[a['linear']]:validate(w['time_warp'],[0,161],[500,661])
 with pytest.raises(ValueError):validate([[0,500],[161,660]],[0,161],[500,661])

def test_interior_warp_changes_core_without_changing_bridge_readset():
 g,_=graph();v=variants(g,4,53);assert v['interior'] is not None
 t=np.arange(161);rv=np.zeros((161,22,3));rv[:,:,2]=(.002*t+.00001*t*t)[:,None];R=Rotation.from_rotvec(rv.reshape(-1,3)).as_matrix().reshape(161,22,3,3)
 def sample(w):w=np.asarray(w['time_warp']);return rotation_sample(R,np.interp(np.arange(500,661),w[:,1],w[:,0]))
 base=sample(v['top'][0]);inside=sample(v['interior']);assert np.max(abs(base-inside))>1e-8;assert np.array_equal(base[np.r_[:4,157:161]],inside[np.r_[:4,157:161]])
 J=np.zeros((22,3));parents=np.r_[-1,np.arange(21)];reference=dict(D2=1e9,D3=1e9,shoulder_wrist_lower_m=0)
 q,e=calculate((base[-4:],base[:4],24,J,parents,reference));qq,ee=calculate((inside[-4:],base[:4],24,J,parents,reference));assert q is not None and np.array_equal(q,qq) and e==ee
 counts=suffix(g);alternative=next(unrank(g,k,counts) for k in range(counts[0][0]) if unrank(g,k,counts)[-2][1]!=v['top'][0]['time_warp'][-2][1]);changed=sample(dict(time_warp=alternative));assert np.array_equal(changed[[0,-1]],base[[0,-1]]) and not np.array_equal(changed[-3:],base[-3:]);different,_=calculate((changed[-4:],base[:4],24,J,parents,reference));assert np.max(abs(different-q))>1e-8

def test_joint_search_keeps_distinct_variants_of_the_same_event():
 nodes={k:dict(heuristic=0.,event_uid=k[0]) for k in ('a0','a1','b0')};pop=SimpleNamespace(nodes=nodes,reference=dict(D3=1));graph=SimpleNamespace(curves={'good':np.zeros(1)},records={'good':dict(selected='G1',methods={'G1':dict(gap_D3=.1)})});edges={(0,'a1','b0'):'good'}
 assert solve(pop,graph,[['a0'],['b0']],edges)['status'].startswith('REJECTED')
 result=solve(pop,graph,[['a0','a1'],['b0']],edges);assert result['nodes']==['a1','b0'] and result['admitted_paths']==1

def test_archive_is_deterministic_without_container_timestamps(tmp_path):
 x={'a':np.arange(5),'b':np.eye(3)};save_arrays(tmp_path/'a.npz',x);save_arrays(tmp_path/'b.npz',dict(reversed(list(x.items()))));assert (tmp_path/'a.npz').read_bytes()==(tmp_path/'b.npz').read_bytes()
