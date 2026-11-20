import numpy as np
from dunhuang.research.hold_timing.metrics import occupancy,excess,low_mask

def test_cell_duration_and_source_bounds_are_not_frame_count():
 h=[{'cells_half_open':[0,120]}];q=np.linspace(0,239,240);a=occupancy(q,h,[0,240])[0]
 assert a['target_seconds']==4 and a['natural_supported_seconds']==4 and a['minimum_seconds_at_rate_075']==3
 b=occupancy(np.linspace(30,269,240),h,[30,270])[0]
 assert b['natural_supported_seconds']==3 and b['clipped_by_core']

def test_same_motion_cost_is_independent_of_music_and_empty_cap_abstains():
 h=[{'target_seconds':4.,'target_frame_centers':[0,120]}]
 assert excess(h,None) is None
 assert excess(h,3.)==1
 assert excess(h,3.,np.zeros(240))<excess(h,3.,np.ones(240))

def test_low_motion_requires_both_rotation_and_fk_not_one_signal():
 s={'angular':np.array([0,.1,0]),'linear':np.array([0,0,.1])}
 assert low_mask(s,{'angular':.05,'linear':.05}).tolist()==[True,False,False]
