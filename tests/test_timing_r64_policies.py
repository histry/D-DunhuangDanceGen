import copy
from dunhuang.research.timing_r64.policies import choose,selection,objective,guard

def row(g=48,o=0,cost=.5,music=.2,guard=True):
    return dict(key=f'x_{g}_{o}',gap_frames=g,timing_offset_frames=o,execution_cache_key=f'{g}_{o}',
        methods=dict(G1=dict(constructible=True,passes_original_screen=True)),
        balanced_cost=dict(balanced_cost=cost),music_reward=dict(primary_music_score=music,section_proxy_score=0.),
        music_reward_shuffled=dict(primary_music_score=1-music,section_proxy_score=0.),retention_guard_passed=guard)

def test_music_changes_reward_only_and_guard_does_not_filter():
    a,b=row(music=.1),row(36,12,music=.9,guard=False)
    before=copy.deepcopy([a,b])
    assert choose([a,b],'LocalBeat') is b
    assert choose([a,b],'ShuffledBeat') is a
    assert [a,b]==before
    selected=selection(dict(key='x',role='EVAL'),[a,b],'LocalBeat')
    assert selected['accepted'] and not selected['retention_guard_passed']

def test_r63_shortest_remains_equal_repayment():
    a,b,c=row(24,12),row(36,0),row(48,0)
    assert choose([a,b,c],'LocalShortest') is b

def test_baselines_not_replaced_when_illegal():
    a=row(24,0);a['methods']['G1']['passes_original_screen']=False
    b=row(36,0)
    assert choose([a,b],'Fixed24') is a
    s=selection(dict(key='x',role='DEV'),[a,b],'Fixed24')
    assert s['constructible'] and not s['accepted'] and s['record_key']==a['key']

def test_unknown_music_is_not_a_perfect_match():
    a=row();a['music_reward']['primary_music_score']=None
    assert objective(a,'LocalBeat') is None and choose([a],'LocalBeat') is None

def test_balanced_equal_is_nested_in_full_lattice():
    a,b=row(cost=.7),row(36,12,cost=.3)
    assert choose([a,b],'BalancedEqual') is a
    assert choose([a,b],'LocalBalanced') is b

def test_relative_guard_is_not_the_original_bridge_gate():
    r=row();r['methods']['G1'].update(passes_original_screen=False,full_dynamics=dict(D3_Q95=1.))
    ratios=dict(FK_path_length_mean_m_ratio=1.,FK_motion_range_rms_m_ratio=1.)
    r.update(left=dict(retention_vs_fixed48=ratios),right=dict(retention_vs_fixed48=ratios))
    assert guard(r,r) and not r['methods']['G1']['passes_original_screen']
