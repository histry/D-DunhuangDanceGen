import copy
from dunhuang.research.timing_r64.study import normalized,relation

def test_replay_ignores_execution_flags_but_not_scientific_values():
    a=dict(binding=dict(commit='x'),records=[dict(prior_bridge_or_core_reused=True,
        execution_cache_lookup=True,independent_e1w_recompile=False,gap_frames=48)])
    b=copy.deepcopy(a);b['binding']['commit']='y'
    b['records'][0].update(prior_bridge_or_core_reused=False,execution_cache_lookup=False,independent_e1w_recompile=True)
    assert normalized(a)==normalized(b)
    b['records'][0]['gap_frames']=24
    assert normalized(a)!=normalized(b)

def test_section_relation_is_proxy_not_phrase_truth():
    r=relation(dict(target_gap_interval=[240,288]),dict(section_proxy_frames=[288]))
    assert r['relation']=='ENDS_AT_WITHIN3_FRAMES'
    assert r['independent_phrase_truth']=='UNKNOWN'

def test_unobserved_section_not_a_perfect_match():
    r=relation(dict(target_gap_interval=[240,288]),dict(section_proxy_frames=[]))
    assert r['status']=='NO_SECTION_PROXY'
