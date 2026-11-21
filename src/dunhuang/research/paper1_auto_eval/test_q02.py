import unittest
import numpy as np
from .q02_common import rank
from .q02_music import select
from .q02_motion import variants
from .music import gaussian_match,motion_beats
from .observability import angular_acc

class Q02Contracts(unittest.TestCase):
    def test_rank_does_not_replace_top1_by_pairwise(self):
        r=rank([0,0,3,0,4]);self.assertEqual(r['paired_win'],.75);self.assertFalse(r['strict_top1']);self.assertEqual(r['tied_top1_credit'],0)
        r=rank([0,0,3,0,3]);self.assertEqual(r['tied_top1_credit'],.5);self.assertFalse(r['strict_top1']);self.assertIsNone(rank([0,None,1,0,0])['strict_top1'])
    def test_known_links_block_new_population(self):
        def src(uid,audio,choreo):return dict(source_uid=uid,audio_sha256=audio,raw_motion_sha256=uid,group=audio,recording_uid=uid,choreography_uid=choreo,dataset='AIST++',role='train')
        meta=dict(sources=[src('a','a','c'),src('b','b','c'),src('d','d','d'),src('e','d','e')]);r=select(meta,dict(sources=[dict(source_uid='a')]))
        self.assertEqual([s['source_uid'] for s in r['sources']],['d']);self.assertFalse(r['independent_generalization_certified'])
    def test_static_has_no_minima_and_is_not_perfect_BAS(self):
        self.assertEqual(len(motion_beats(np.zeros((120,22,3)))),0);self.assertIsNone(gaussian_match([1,2],[]))
    def test_wrist_dose_protects_edges_and_is_observable(self):
        R=np.tile(np.eye(3),(120,22,1,1));c=dict(unit=dict(source_interval=[0,120]),music_start_frame=0,hold=dict(cells_half_open=[30,50]))
        # Exercise only fresh noise variants; old legal clock generation needs full source metadata.
        from unittest.mock import patch
        with patch('dunhuang.research.paper1_auto_eval.q02_motion.controls.perturbations',return_value=iter([])):
            rows=list(variants(R,c,np.arange(.25,4,.5)))
        noise=[v for v in rows if v[0].startswith('WristNoise')]
        self.assertEqual(len(noise),3);values=[]
        for label,a,q,mask,scope,dose in noise:
            self.assertTrue(np.array_equal(a[~mask],R[~mask]));values.append(angular_acc(a).max())
        self.assertTrue(np.all(np.diff(values)>0))
if __name__=='__main__':unittest.main()
