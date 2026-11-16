"""CPU-only input checks shared with fresh verification."""
import numpy as np
from .clock import source_positions
from dunhuang.research.complete_r36.core import components

def verify_plan(plan,lib):
    assert plan['events'][0]['music_interval'][0]==0 and plan['events'][-1]['music_interval'][1]==plan['frames']
    assert all(a['music_interval'][1]==b['music_interval'][0] for a,b in zip(plan['events'],plan['events'][1:]))
    for e in plan['events']:
        q=source_positions(e,np.arange(*e['music_interval']));a,b=e['source_interval']
        assert q.min()>=a and q.max()<=b-1
        assert any(lo<=a<b<=hi for lo,hi in components([r['source_interval'] for r in lib.rows[e['source_uid']]]))
