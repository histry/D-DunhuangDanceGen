import numpy as np
import pytest
from dunhuang.research.music_structure_r56.common import CFG,kernel
from dunhuang.research.music_structure_r56.music import novelty,descriptors
from dunhuang.research.music_structure_r56.slots import allocate,shuffled,centers,coverage
from dunhuang.research.music_structure_r56.metrics import correspondence
from dunhuang.research.music_structure_r56.planning import choose

def test_exact_duration_and_transition_clock():
 spans,_=allocate(3601,np.zeros(3601),range(180,301));assert np.all(coverage(spans,3601)==1);assert len(centers(spans))==12;assert sum(r['kind']=='core' for r in spans)==13;assert all(r['interval'][1]-r['interval'][0]==48 for r in spans if r['kind']=='transition')
 with pytest.raises(ValueError):allocate(100,np.zeros(100),range(180,301))

def test_music_changes_layout_without_changing_budget():
 baseline,_=allocate(3601,np.zeros(3601),range(180,301));signal=np.zeros(3601);signal[np.asarray(centers(baseline))+30]=10;candidate,_=allocate(3601,signal,range(180,301));assert centers(candidate)!=centers(baseline);assert all(180<=r['interval'][1]-r['interval'][0]<=300 for r in candidate if r['kind']=='core')

def test_shuffled_structure_preserves_duration_multiset():
 spans,_=allocate(3601,np.linspace(0,1,3601),range(180,301));other,info=shuffled(spans,0);get=lambda ss:sorted(r['interval'][1]-r['interval'][0] for r in ss if r['kind']=='core');assert get(spans)==get(other) and info['same_duration_multiset'];assert np.all(coverage(other,3601)==1)

def test_proposal_does_not_need_motion_or_predicted_alignment():
 x=np.zeros((600,4));x[:300,0]=1;x[300:,1]=1;v=novelty(x);assert abs(np.argmax(v)-300)<10 and np.isfinite(v).all();assert not novelty(np.zeros_like(x)).any()

def test_proxy_correspondence_is_unique_not_many_to_one():
 r=correspondence([10,11],[10],3);assert r['matches']==1 and r['precision']==.5 and r['recall']==1;assert correspondence([],[],3)['F1'] is None

def test_structure_search_cannot_override_rejected_geometry():
 node=lambda u:{'unit':{'unit_uid':u,'source_uid':u},'bank':int(u),'heuristic':0}
 pools=[[node('0')],[node('1')]];sim=np.eye(2);choice,_=choose(pools,{}, {'D3':1},sim,sim);assert choice is None

def test_kernel_preserves_equal_items_and_phase_difference():
 x=np.array([[1,2,3],[1,2,3],[3,2,1]]);k=kernel(x);assert k[0,1]==1 and k[0,2]<1 and np.array_equal(k,k.T)
