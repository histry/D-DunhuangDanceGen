from fractions import Fraction
import numpy as np
import pytest
from dunhuang.planning.feasible_time_map import domain,solve
from dunhuang.planning.time_map import source_positions
from dunhuang.data.change_event_audit import slot_eligibility

def brute(a,b,A,B,c):
    if c is None or c-12<a or c+12>=b or c-12==a:return []
    if not 75*(b-a)<=100*(B-A)<=133*(b-a):return []
    answer=[]
    for t in range(A+12,B-12):
        ds=[c-12-a,24,b-c-12];dt=[t-12-A,24,B-t-12]
        if all(s>0 and v>0 and 75*s<=100*v<=133*s for s,v in zip(ds,dt)):answer.append(t)
    return answer

def entry(a,b,c,support=True):return dict(source_interval=[a,b],center=c,normalization_support=support)

def test_integer_domain_and_choice_equal_exhaustive_oracle():
    rng=np.random.default_rng(20261112)
    for _ in range(600):
        a,A=map(int,rng.integers(0,1000,2));n,d=map(int,rng.integers(25,201,2));c=a+int(rng.integers(0,n))
        oracle=brute(a,a+n,A,A+d,c);result=domain([a,a+n],[A,A+d],c)
        actual=[] if result["integer_domain"] is None else list(range(result["integer_domain"][0],result["integer_domain"][1]+1))
        assert actual==oracle
        beats=[A+int(t) for t in rng.integers(0,d,5)];r=solve(entry(a,a+n,c),dict(start=A,end=A+d),beats)
        if oracle:
            preferred=A+round(Fraction((c-a)*d,n))
            assert r["tick"]==min(oracle,key=lambda t:(min(abs(t-b) for b in beats),abs(t-preferred),t))
        else:assert r["tick"] is None

def test_real_25_frame_window_does_not_make_zero_length_segment_legal():
    r=solve(entry(100,190,112),dict(start=0,end=90),[12])
    assert r["status"]=="ZERO_OR_NONMONOTONE_WARP"
    assert r["integer_domain"] is None and r["time_warp"] is None

def test_all_beats_illegal_selects_nearest_feasible_integer():
    r=solve(entry(0,100,20),dict(start=0,end=100),[35])
    assert r["status"]=="PASS" and r["tick"]==22
    assert r["beat_distance_frames"]==13
    assert slot_eligibility(entry(0,100,20),dict(start=0,end=100),[24])[0] is False

def test_legal_beat_preferred_then_linear_then_earlier():
    r=solve(entry(0,100,50),dict(start=0,end=100),[49,51,51])
    assert r["tick"]==49 and r["feasible_beat_count"]==2

def test_no_slot_beats_use_linear_projection_with_no_fake_distance():
    r=solve(entry(0,100,50),dict(start=100,end=200),[99,201])
    assert r["tick"]==150 and r["beat_distance_frames"] is None

def test_exact_boundary_rates_and_duration_rejections():
    for duration in (75,133):
        r=solve(entry(0,100,50),dict(start=0,end=duration),[])
        assert (r["status"]=="PASS")==bool(brute(0,100,0,duration,50))
    for duration in (74,134):
        assert solve(entry(0,100,50),dict(start=0,end=duration),[])["status"]=="DURATION_RATIO"

def test_forward_inverse_and_phase_order_use_unchanged_v1_operator():
    r=solve(entry(100,220,160),dict(start=0,end=140),[70])
    source=np.array([100,111,135,160,180,219],float);w=np.array(r["time_warp"])
    times=np.interp(source,w[:,0],w[:,1]);assert np.all(np.diff(times)>0)
    result=source_positions(dict(source_interval=[100,220],music_interval=[0,140],time_warp=r["time_warp"]),times)
    np.testing.assert_allclose(result,source,atol=1e-10)

def test_masks_do_not_turn_unknown_into_a_time_map():
    assert solve(entry(0,100,None),dict(start=0,end=100),[])["status"]=="NO_SAFE_ANCHOR"
    assert solve(entry(0,100,50,False),dict(start=0,end=100),[])["status"]=="NORMALIZER_SUPPORT"
    assert solve(entry(0,100,50,False),dict(start=0,end=100),[],mode="adapter")["status"]=="PASS"

@pytest.mark.parametrize("args",[
    ([-1,100],[0,100],50),([0,0],[0,100],0),([0,100],[0,100],True),
    ([0,100],[0,100],float("nan")),([0,100],[0,100],50.0)])
def test_invalid_physical_coordinates_fail_closed(args):
    with pytest.raises(ValueError):domain(*args)

def test_invalid_beat_and_mode_fail_closed():
    for beats in ([True],[float("nan")],[-1]):
        with pytest.raises(ValueError):solve(entry(0,100,50),dict(start=0,end=100),beats)
    with pytest.raises(ValueError):solve(entry(0,100,50),dict(start=0,end=100),[],"unknown")

def test_old_accepted_maps_remain_in_complete_feasible_domain():
    for c in range(13,87):
        for length in range(75,134):
            e=entry(0,100,c);slot=dict(start=0,end=length)
            good,_,warp=slot_eligibility(e,slot,[30,60,90])
            if good:
                d=domain([0,100],[0,length],c)
                assert d["status"]=="PASS"
                assert d["integer_domain"][0]<=warp[1][1]+12<=d["integer_domain"][1]
