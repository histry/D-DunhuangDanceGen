"""Scientific wiring checks; synthetic fixtures are not quality evidence."""
import numpy as np
import pytest
import json
from dunhuang.research.retiming_baselines_r69 import study as s

def fixture():
 R=np.broadcast_to(np.eye(3),(240,22,3,3)).copy();q=np.arange(240,dtype=float)
 full=np.broadcast_to(np.eye(3),(528,22,3,3)).copy()
 case=dict(source_interval=[0,240],parent_side='left',parent_record_key='fixed')
 connection=dict(candidate_R_sha256=s.array_hash(R),full_R_sha256=s.array_hash(full),bridge_R_sha256=s.array_hash(full[240:288]),parent_record_key='fixed',gap_frames=48)
 return case,q,R,connection,full

def test_saved_execution_has_actual_source_and_unchanged_full_core():
 c,q,R,receipt,full=fixture()
 assert s.verify_execution(c,q,R,receipt,full,R)
 wrong=dict(receipt,parent_record_key='another')
 with pytest.raises(ValueError,match='neighbor'):s.verify_execution(c,q,R,wrong,full,R)
 q[100]=q[99]
 with pytest.raises(ValueError,match='monotone'):s.verify_execution(c,q,R,receipt,full,R)

def test_saved_bridge_hash_is_required_even_for_legal_clock():
 c,q,R,receipt,full=fixture();receipt['bridge_R_sha256']='0'*64
 with pytest.raises(ValueError,match='Bridge'):s.verify_execution(c,q,R,receipt,full,R)

def test_raw_ties_keep_registered_order_and_do_not_use_geometry():
 rows=[dict(scores=dict(Single64=v),connectivity=dict(passes_original_screen=False)) for v in [1.,1.,None,.5]]
 assert s.raw_ties(rows,'Single64')==[0,1]
 assert s.raw_ties([dict(scores=dict(Single64=None))],'Single64')==[]
 with pytest.raises(ValueError):s.raw_ties([dict(scores=dict(Single64=float('nan')))],'Single64')

def test_failed_new_solver_does_not_construct_bridge_or_replace_case(monkeypatch):
 monkeypatch.setattr(s.solver,'beat_dp',lambda *a,**k:dict(status='NO_EVENT',reason='SOURCE_HAS_NO_EVENT',runtime_seconds=.1))
 monkeypatch.setattr(s.connectivity,'check',lambda *a:pytest.fail('No output may be bridged'))
 case=dict(source_interval=[0,240],target_interval=[1000,1240],phase_inputs=[])
 value,cost=s.proposed(case,'BeatDP',None,dict(beat_frames=[]),{})
 assert value['result']['status']=='NO_EVENT' and value['R'] is None and cost==.1

def test_summary_keeps_unavailable_rows_and_does_not_report_quality_winner():
 rows=[dict(method=m,status='NO_EVENT',selection={}) for m in s.CFG['methods'] for _ in range(12)]
 result=s.summary(rows)
 assert result['registered_method_positions']==72 and result['human_review_records']==0
 assert all(r['registered_cases']==12 and r['quality_winner'] is None for r in result['methods'].values())
 assert not result['can_route'] and result['artistic_quality']=='NOT_EVALUABLE'

def test_new_legal_clock_is_serializable_and_forces_real_geometry_call(monkeypatch):
 c,q,R,receipt,full=fixture();c.update(target_interval=[1000,1240],phase_inputs=[])
 monkeypatch.setattr(s.solver,'beat_dp',lambda *a,**k:dict(status='LEGAL',reason=None,q=q.copy(),source_positions=q.copy(),time_warp=[[0,1000],[239,1239]],runtime_seconds=.1))
 calls=[]
 def check(case,body,ctx):
  calls.append(ctx['cache'].copy());return receipt,dict(full_R=full)
 monkeypatch.setattr(s.connectivity,'check',check)
 v,wall=s.proposed(c,'BeatDP',R,dict(beat_frames=[]),dict(cache={'stale':'must not reuse'}))
 assert calls==[{}] and wall==.1 and 'q' not in v['result'] and 'source_positions' not in v['result']
 json.dumps(v['result']);np.testing.assert_array_equal(v['q'],q)
