import numpy as np
import pytest
from dunhuang.research.timing_validity_r66 import population
from dunhuang.research.timing_validity_r66.study import donor_distance,energy_score,synthetic_control

def test_nonperiodic_exact_event_clock():
    assert synthetic_control()['status']=='PASS'

def test_energy_sign_and_nonwrap_support():
    x=np.sin(np.arange(120)*.37)
    clock={'energy_curve':np.r_[np.zeros(20),x,np.zeros(20)]}
    speed=clock['energy_curve'].copy()
    assert energy_score(speed,clock,[20,140],0)==pytest.approx(1)
    assert energy_score(speed,clock,[20,140],8)<1
    assert energy_score(speed,clock,[0,120],-8) is None

def test_donor_requires_local_beat_and_five_offset_support():
    clock={'energy_curve':[1.]*200,'beat_frames':list(range(5,200,15))}
    assert donor_distance(clock,clock,[20,140])==pytest.approx(0)
    short=dict(clock,energy_curve=[1.]*145)
    assert donor_distance(clock,short,[20,140]) is None
    no_local=dict(clock,beat_frames=[0,150,190])
    assert donor_distance(clock,no_local,[20,140]) is None

def test_population_forbids_test_or_offset_redesign():
    with pytest.raises(ValueError):population.select({},dict(offsets=[0,15]))
