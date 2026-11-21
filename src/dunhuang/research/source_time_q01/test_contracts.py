import unittest
from . import baselines as b

def row(uid,source,error,gtw=1,support=4,ok=True):return dict(uid=uid,source_order=source,anchor_target_frame=error,clock_legal=True,moving_support_seconds=support,G1_pass=ok,normalized_clock_RMS=.1,GTWLocal=gtw,labels=['Uniform'])
class Contracts(unittest.TestCase):
    def test_strong_fallback_checks_next_source(self):
        rows=[row('a',0,8,support=6),row('b',1,0,support=4)];d=b.choose('SequentialTargetFallback',rows,0);self.assertEqual(d['candidate_uid'],'b');self.assertTrue(d['success']);self.assertEqual(d['full_pool_budget'],2)
    def test_cross_source_gtw_cannot_select_joint(self):
        rows=[row('a',0,1,gtw=1000),row('b',1,2,gtw=.00001)];self.assertEqual(b.choose('Joint',rows,0)['candidate_uid'],'a')
    def test_exhaustive_sequential_same_as_joint(self):
        rows=[row('a',0,1),row('b',1,0),row('c',1,2)];self.assertEqual(b.choose('Joint',rows,0)['candidate_uid'],b.choose('ExhaustiveSequential',rows,0)['candidate_uid'])
    def test_failed_target_not_replaced_by_diagnostic(self):
        rows=[row('a',0,9),row('b',1,10)];d=b.choose('Joint',rows,0);self.assertIsNone(d['candidate_uid']);self.assertIsNotNone(d['diagnostic_uid']);self.assertFalse(d['success'])
    def test_numerical_failure_is_not_hidden_by_clock_score(self):
        rows=[row('a',0,0,ok=False),row('b',1,2)];self.assertEqual(b.choose('Joint',rows,0)['candidate_uid'],'b')
    def test_target_tolerance_and_support_both_required(self):
        self.assertTrue(b.feasible(row('a',0,3),0));self.assertFalse(b.feasible(row('a',0,3.01),0));self.assertFalse(b.feasible(row('a',0,0,support=1),0))
if __name__=='__main__':unittest.main()
