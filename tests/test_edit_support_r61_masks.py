import importlib.util
from pathlib import Path

import numpy as np
import pytest


# The module is deliberately pure and can be verified before the research
# package is installed or authenticated experiment assets are mounted.
MODULE = Path(__file__).resolve().parents[1] / 'src/dunhuang/research/edit_support_r61/mask_audit.py'
spec = importlib.util.spec_from_file_location('r61_mask_audit_under_test', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
audit_mask = module.audit_mask


def test_fractional_reads_use_both_floor_and_ceil_not_nearest_rounding():
    r = audit_mask([9.1, 9.2, 10.0], [True, False, False], [[10, 11]], [[8, 13]], [10])
    assert r['source_read_support'] == dict(floor=[9, 9, 10], ceil=[10, 10, 10])
    assert r['raw']['target_mask'] == [True, True, True]
    assert r['raw']['edit_overlap_target_frames'] == 1
    assert r['raw']['edited_source_frames'] == [10]
    assert r['edited_centers'] == [10]
    assert r['raw']['source_intrusion_fraction'] == 1.0


def test_unknown_verified_support_is_not_a_zero_overlap_measurement():
    r = audit_mask([0, 1, 2], [False, True, False], [[1, 2]], [[0, 3]], [1])
    assert r['verified_status'] == 'UNKNOWN'
    assert r['verified'] is None and r['verified_intrusion_fraction'] is None
    assert r['task_status'] == 'RECONSTRUCTION'
    assert r['preserved_source_claim_authorized'] is False


def test_noedit_retains_targets_and_does_not_touch_phase_centers():
    r = audit_mask([2, 3, 4], [False] * 3, [[2, 4]], [[1, 6]], [2, 3])
    assert r['edit_target_frames'] == 0
    assert r['retained_target_mask'] == [True] * 3
    assert r['raw']['edit_overlap_target_frames'] == 0
    assert r['raw']['source_intrusion_fraction'] == 0.0
    assert r['edited_centers'] == []
    assert r['task_status'] == 'READ_ONLY_SOURCE'


def test_overlapping_spans_are_unioned_not_double_counted():
    r = audit_mask([0, 1, 2, 3, 4], [False, True, True, False, False],
                   [[1, 3], [2, 4], [1, 3]], [[0, 5]], [1, 2], [[1, 3]])
    assert r['raw']['source_spans'] == [[1, 4]]
    assert r['raw']['source_support_frame_count'] == 3
    assert r['raw']['edited_source_frame_count'] == 2
    assert r['raw']['source_intrusion_fraction'] == pytest.approx(2 / 3)
    assert r['verified_intrusion_fraction'] == 1.0


def test_half_open_end_and_integer_frame_support():
    r = audit_mask([3, 4, 5], [False, True, True], [[3, 4]], [[3, 5]], [3, 4])
    assert r['raw']['target_mask'] == [True, False, False]
    assert r['raw']['edit_overlap_target_frames'] == 0
    assert r['band']['edit_overlap_target_frames'] == 1
    assert r['edited_source_read_frames'] == [4, 5]
    assert r['edited_centers'] == [4]


def test_empty_evidence_is_no_evidence_not_semantic_protection_success():
    r = audit_mask([0, 1], [True, False], [], [], [])
    assert r['raw']['evidence_status'] == 'NO_EVIDENCE'
    assert r['raw']['source_intrusion_fraction'] is None
    assert r['verified_status'] == 'UNKNOWN'


def test_unsigned_positions_cannot_hide_descending_order_by_underflow():
    with pytest.raises(ValueError):
        audit_mask(np.array([2, 1], dtype=np.uint64), [False, True], [], [], [])


@pytest.mark.parametrize('positions', [[], [1, 1], [2, 1], [-1, 0], [0, np.nan],
                                       [0, np.inf], [[0, 1]], [True, False], ['0', '1']])
def test_rejects_invalid_source_positions(positions):
    with pytest.raises(ValueError):
        audit_mask(positions, [False, False], [], [], [])


@pytest.mark.parametrize('edit', [[0, 1], [True], [[False, True]], ['false', 'true']])
def test_rejects_non_boolean_or_wrong_length_edit_mask(edit):
    with pytest.raises(ValueError):
        audit_mask([0, 1], edit, [], [], [])


@pytest.mark.parametrize('spans', [[[1, 1]], [[2, 1]], [[-1, 2]], [[1.0, 2]],
                                   [[True, 2]], [[1, 2, 3]], [[1]], None])
def test_rejects_invalid_evidence_intervals(spans):
    with pytest.raises(ValueError):
        audit_mask([0, 1], [False, True], spans, [], [])


@pytest.mark.parametrize('centers', [[1, 1], [-1], [1.0], [True], None])
def test_rejects_invalid_centers(centers):
    with pytest.raises(ValueError):
        audit_mask([0, 1], [False, True], [], [], centers)
