import numpy as np,pytest
from dunhuang.research.execution_planning_r58.common import execution_cost,regions_for
from dunhuang.research.execution_planning_r58.graph import boundary_lattice,choose,path_capacity
def test_execution_cost_penalizes_amplification_and_suppression_symmetrically():
 assert execution_cost({'1':{'ratio':2.}})==pytest.approx(execution_cost({'1':{'ratio':.5}}));assert execution_cost({'3':{'ratio':None}})==0;assert execution_cost({'1':{'ratio':0.}})>execution_cost({'1':{'ratio':1.}})
def test_lattice_contains_fixed_plan_and_only_registered_offsets():
 base=[dict(kind='core',interval=[0,240]),dict(kind='transition',interval=[240,288]),dict(kind='core',interval=[288,528])];q=boundary_lattice(base);assert [0,240] in q[0] and [288,528] in q[1];assert sorted(b for a,b in q[0])==[222,240,258]
def test_continuous_budget_is_checked_in_real_clock():
 n=[dict(target=[0,200]),dict(target=[248,500])];r=regions_for(n);assert r[1]['interval']==[200,248]
 with pytest.raises(ValueError):regions_for([n[0],dict(target=[250,500])])
def test_joint_search_uses_compatible_edges_and_protects_same_incumbent(monkeypatch):
 import dunhuang.research.execution_planning_r58.graph as g
 monkeypatch.setitem(g.CFG,'events',2);monkeypatch.setitem(g.CFG,'gap',2);monkeypatch.setitem(g.CFG,'min_core',4);monkeypatch.setitem(g.CFG,'max_core',10)
 def node(key,target,src):return dict(id=key,target=target,event_uid=key,source_uid=src,heuristic=0.,execution_cost=0.)
 evidence=dict(selected='G1',methods=dict(G1=dict(gap_D3=1.)))
 graph=dict(layers=[[node('a',[0,6],'left'),node('b',[0,8],'left')],[node('c',[8,16],'right'),node('d',[10,16],'right')]],edges=[dict(left='a',right='c',eligible=True,evidence=evidence),dict(left='b',right='d',eligible=True,evidence=evidence)],base_intervals=[[0,6],[8,16]])
 reward=np.zeros(20);reward[9]=1.;inc,_=choose(graph,reward,dict(D3=1),fixed=True);joint,_=choose(graph,reward,dict(D3=1),incumbent=inc);assert joint['ids']==['b','d'];assert path_capacity(graph)=='2';assert path_capacity(graph,True)=='1'
def test_native_unknown_channels_and_absolute_target_identity():
 from dunhuang.research.execution_planning_r58.profiles import Assets
 from dunhuang.research.tempo_attribution_r57.common import stats
 from types import SimpleNamespace
 a=object.__new__(Assets);R=np.tile(np.eye(3),(240,22,1,1));a.pop=SimpleNamespace(lib=SimpleNamespace(rotations={'s':R},rows={'s':[dict(reference=dict(sha256='raw'))]}),net=SimpleNamespace(joints=np.zeros((22,3)),parents=np.array([-1]+[0]*21)));a.native={'e':stats(np.zeros((240,22,3)))};u=dict(unit_uid='e',source_uid='s',source_interval=[0,240],phases=[]);track=dict(structure=dict(beats=[],energy_percentile=np.zeros(600)))
 assert a.node(u,[0,100],track) is None
 first,_=a.node(u,[0,240],track);second,_=a.node(u,[300,540],track);assert first['id']!=second['id'];assert second['warp']['time_warp'][0][1]==300;assert first['world_root_contact']=='UNKNOWN';assert first['source_normalized']['3']['ratio'] is None
