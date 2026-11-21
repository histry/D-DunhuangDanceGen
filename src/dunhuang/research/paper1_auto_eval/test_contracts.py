import unittest
import numpy as np
from .music import gaussian_match,bind,scores,motion_beats
from .reviews import classification,summarize

class Contracts(unittest.TestCase):
    def test_roundoff_identity_and_unavailability(self):
        a=np.zeros((4,22,3,3));self.assertEqual(classification(a,a+1e-12),'VALID_IDENTICAL');self.assertEqual(classification(a,a+1e-5),'VALID_DISTINCT');self.assertEqual(classification(a,None),'UNAVAILABLE')
    def test_BAS_is_directional_and_missing_is_not_zero_quality(self):
        self.assertIsNone(gaussian_match([], [1]));self.assertGreater(gaussian_match([0],[0,1]),gaussian_match([0,1],[0]));self.assertEqual(len(motion_beats(np.zeros((30,22,3)))),0)
    def test_binding_retains_seconds_even_when_rounding_exits_support(self):
        c=dict(fps=30,frames=100,audio_sha256='x',model_manifest_sha256='y',events=[dict(uid='x',seconds_supported=True,seconds=19.9/30)])
        b=bind(c,0,[10,20]);self.assertAlmostEqual(b['events'][0]['local_frame_float'],9.9);self.assertFalse(b['events'][0]['nearest_local_frame_supported']);self.assertFalse(b['used_for_selection'])
    def test_offsets_keep_music_queries_without_wrap(self):
        c=dict(events=[dict(source='beat_this',kind='beat',seconds_supported=True,seconds=t) for t in [.3,.8,1.3,1.8]])
        rows=[r for r in scores([.4,1.0,1.5],c,c,0,2) if r['stream']=='beat_this:beat' and r['condition']!='wrong_song'];self.assertEqual({r['music_events'] for r in rows},{2});self.assertEqual(len(rows),5)
    def test_identical_preferences_are_excluded_from_effect_states(self):
        admin=dict(pack_id='TEST',items=[dict(id='d',classification='VALID_DISTINCT',control=False,available=True,positive_side='A',contrast='TEST',source_uid='s'),dict(id='i',classification='VALID_IDENTICAL',control=True,available=True,positive_side=None,different_expected='NO',source_uid='s')])
        doc=dict(schema='RETIME_PREF_V1',pack_id='TEST',task='naturalness',review_id='TEST',reviewer_id='TEST',reviewer_roster_ref='TEST',reviewed_at='2026-11-21T00:00:00+08:00',independence=dict(own_judgment=True),exposure=dict(method_identity_seen='NO',prior_named_results_seen='NO',other_reviews_seen='NO'),items=[dict(id='d',choice='A',different='YES'),dict(id='i',choice='B',different='NO')])
        result=summarize([doc],admin);self.assertEqual(result['states'],{'A':1});self.assertEqual(result['identical_control_choices'],{'B':1});self.assertEqual(result['quality'],'NOT_EVALUABLE')
if __name__=='__main__':unittest.main()
