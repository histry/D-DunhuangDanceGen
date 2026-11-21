import importlib.util
import unittest
from pathlib import Path

path=Path(__file__).resolve().parents[1]/'scripts/diagnose_fgmmc.py'
spec=importlib.util.spec_from_file_location('fgmmc_frozen_diagnosis',path)
diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)

class DiagnosisContracts(unittest.TestCase):
    def test_zero_reference_retains_all_tokens(self):
        audit=diag.structural_audit();r=next(r for r in audit['rows'] if r['offset_frames']==0)
        self.assertEqual(r['unrestricted_reference_matches'],16)
        self.assertEqual(r['uniform_clock_matches'],16)
        self.assertEqual(r['best_finite_clock_matches'],16)

    def test_eight_frame_shift_can_be_partly_absorbed(self):
        audit=diag.structural_audit()
        for delta in [-8,8]:
            r=next(r for r in audit['rows'] if r['offset_frames']==delta)
            self.assertEqual(r['uniform_clock_matches'],0)
            self.assertGreater(r['best_finite_clock_matches'],0)
            self.assertLessEqual(r['best_finite_clock_matches'],r['unrestricted_reference_matches'])

    def test_fifteen_frame_shift_is_not_equivalent_to_eight(self):
        audit=diag.structural_audit()
        for sign in [-1,1]:
            a=next(r for r in audit['rows'] if r['offset_frames']==8*sign)
            b=next(r for r in audit['rows'] if r['offset_frames']==15*sign)
            self.assertLess(b['finite_over_unrestricted'],a['finite_over_unrestricted'])

    def test_music_crop_sign_uses_original_time(self):
        centers,_,_,_=diag.geometry();bounds=diag.structural_audit()['actual_pool_bounds']
        positive=diag.path_statistics([(0,1)],centers,8,bounds)
        negative=diag.path_statistics([(0,1)],centers,-8,bounds)
        self.assertLess(positive['physical_mean_gap_frames'],negative['physical_mean_gap_frames'])
        self.assertGreater(positive['pooled_support_overlap'],negative['pooled_support_overlap'])

    def test_missing_time_correspondence_stays_unknown(self):
        centers,_,_,_=diag.geometry();bounds=diag.structural_audit()['actual_pool_bounds']
        self.assertIsNone(diag.path_statistics([],centers,8,bounds)['physical_mean_gap_frames'])
        self.assertIsNone(diag.path_statistics([(0,0)],centers,None,bounds)['pooled_support_overlap'])

    def test_existing_clock_population_is_unchanged(self):
        audit=diag.structural_audit()
        self.assertEqual(sorted(r['amplitude'] for r in audit['clock_order']),[-8,-4,0,4,8])
        self.assertTrue(audit['network_masks_same_across_offset_labels'])

if __name__=='__main__':unittest.main()
