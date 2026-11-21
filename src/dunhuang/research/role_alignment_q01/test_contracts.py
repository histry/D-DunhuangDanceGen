import unittest
import numpy as np
from .alignment import score,shuffled,options
from .events import select_nodes
class Contracts(unittest.TestCase):
    def test_empty_evidence_not_fake_quality(self):self.assertIsNone(score([],[]));self.assertIsNone(score([dict(uid='q',time=1,role='SPEED_PEAK')],[]))
    def test_type_information_changes_pairing(self):
        q=[dict(uid='q',time=1,role='SPEED_PEAK')];m=[dict(uid='m',time=1,role='BEAT')];self.assertAlmostEqual(score(q,m),0);self.assertAlmostEqual(score(q,m,False),1)
    def test_distinct_ordered_music_ids(self):
        q=[dict(uid=str(i),time=1,role='SPEED_PEAK') for i in range(3)];m=[dict(uid='m',time=1,role='ACCENT_PROXY')];self.assertAlmostEqual(score(q,m),1/3)
    def test_clock_sign_is_event_shift_for_all_methods(self):
        q=[dict(uid='q',time=1,role='ACCEL_RISE')];m=[dict(uid='m',time=1,role='BEAT')];self.assertGreater(score(q,m),score(q,[dict(m[0],time=1.5)]))
    def test_shuffle_preserves_times_ids_and_total_count(self):
        m=[dict(uid='m'+str(i),time=i,role=r) for i,r in enumerate(['BEAT','DOWNBEAT','ACCENT_PROXY','STRUCTURE_PROXY'])];s=shuffled(m);self.assertEqual([(r['uid'],r['time']) for r in s],[(r['uid'],r['time']) for r in m]);self.assertNotEqual([r['role'] for r in s],[r['role'] for r in m])
    def test_nodes_allow_missing_roles(self):
        ev=[dict(uid='a',frame=20,evidence_span=[19,22],role='SPEED_PEAK',supported=True)];self.assertEqual(len(select_nodes(ev,[0,100])),1)
    def test_proposals_frozen_before_feasibility(self):
        n=[dict(uid='n',frame=50,role='SPEED_PEAK')];m=[dict(uid=str(i),time=t,role='ACCENT_PROXY' if i%2 else 'BEAT') for i,t in enumerate([1.4,1.5,1.6,1.7,3.9])];o=options(n,m,[0,101],101,0);self.assertLessEqual(len(o[0]),4);self.assertTrue(all(abs(v['local_frame']-50)<=15 for v in o[0]));self.assertFalse(any(v['time']==3.9 for v in o[0]))
if __name__=='__main__':unittest.main()
