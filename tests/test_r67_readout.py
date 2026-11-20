"""TEST_FIXTURE only: clock/mask/readout arithmetic, no real quality labels."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

from dunhuang.research.sdmr_warp_rank import readout as r


def fixture(n=128):
    tokens = np.tile([1., 0., 0.], (n, 1))
    valid = np.ones(n, dtype=bool)
    return tokens, valid


def test_constant_tokens_all_methods_and_candidates_equal():
    tokens, valid = fixture()
    result = r.score_bank(tokens, [tokens, tokens * 7], valid, [valid, valid])
    for values in result["scores"].values():
        np.testing.assert_allclose(values, [1., 1.], atol=1e-14)
    assert result["trained_parameters"] == 0


def test_mask_holes_keep_physical_bins_and_global_bank_intersection():
    tokens, valid = fixture()
    missing = valid.copy(); missing[17:21] = False
    result = r.score_bank(tokens, [tokens, tokens], valid, [valid, missing])
    common = np.asarray(result["support"]["common_valid_mask"])
    assert common.sum() == 124 and not common[17:21].any()
    assert result["scales"]["Single64"]["cell_intervals_half_open"][0] == [-.5, 1.5]
    assert result["scores"]["Single64"] == [None, None]
    for scale in result["scales"].values():
        np.testing.assert_allclose(scale["per_token_total_pool_weight"], common.astype(float))


def test_overlap_corrected_and_last_frame_retained():
    tokens, valid = fixture(256)
    result = r.score_bank(tokens, [tokens], valid, [valid])
    coarse = result["scales"]["Coarse16"]
    assert max(coarse["per_token_window_coverage"]) == 2
    assert coarse["sparse_pool_weights"][-1]["indices"][-1] == 255
    assert coarse["cell_intervals_half_open"][-1][1] == 255.5
    np.testing.assert_allclose(coarse["per_token_total_pool_weight"], 1.)
    assert sum(coarse["cell_mass"]) == 256
    assert sum(coarse["cell_score_weight"]) == pytest.approx(1.)


def test_pool_then_normalize_matches_explicit_cosine():
    tokens, valid = fixture()
    motion = np.tile([0., 2., 0.], (128, 1))
    result = r.score_bank(tokens, [motion], valid, [valid])
    for values in result["scores"].values():
        assert values == pytest.approx([0.])
    motion = np.tile([3., 4., 0.], (128, 1))
    result = r.score_bank(tokens, [motion], valid, [valid])
    for values in result["scores"].values():
        assert values == pytest.approx([.6])


def test_near_zero_pool_is_missing_not_zero_cosine():
    tokens, valid = fixture()
    zero = np.zeros_like(tokens)
    result = r.score_bank(tokens, [zero, tokens], valid, [valid, valid])
    for method in r.METHODS:
        assert result["scores"][method][0] is None
        assert result["scores"][method][1] == pytest.approx(1.)
        assert "NORM_TOO_SMALL" in str(result["candidate_status"][method][0]["reasons"])


def test_fewer_than_64_valid_centres_retains_empty_bin_failure():
    tokens, valid = fixture(63)
    result = r.score_bank(tokens, [tokens], valid, [valid])
    assert result["support"]["common_valid_count"] == 63
    assert result["scores"]["Single64"] == [None]
    assert result["scales"]["Single64"]["empty_bin_indices"]
    assert result["scores"]["Multi64"] == pytest.approx([1.])


def test_fixed_support_reused_exactly_and_not_silently_intersected():
    tokens, valid = fixture()
    fixed = valid.copy(); fixed[10] = False
    result = r.score_bank(tokens, [tokens], valid, [valid], fixed_support=fixed)
    assert result["support"]["common_valid_mask"] == fixed.tolist()
    assert result["support"]["rule"] == "SUPPLIED_FIXED_SUPPORT_UNCHANGED"
    actual = valid.copy(); actual[11] = False
    with pytest.raises(ValueError, match="no silent intersection"):
        r.score_bank(tokens, [tokens], actual, [valid], fixed_support=fixed)


def test_invalid_input_and_explicit_centres_preserve_clock():
    tokens, valid = fixture()
    centres = np.arange(128, dtype=float) + 6
    result = r.score_bank(tokens, [tokens], valid, [valid], centres=centres)
    assert result["scales"]["Single64"]["cell_intervals_half_open"][0] == [5.5, 7.5]
    assert result["method_components"]["UniformReadout64"][1:] == [
        {"scale": "Mid16", "weight": .25}, {"scale": "Mid16", "weight": .25}]
    for bad in (np.zeros(128), np.arange(127), np.r_[np.arange(127), np.nan]):
        with pytest.raises(ValueError):
            r.score_bank(tokens, [tokens], valid, [valid], centres=bad)
    with pytest.raises(ValueError):
        r.score_bank(tokens, [tokens], valid.astype(int), [valid])
    with pytest.raises(ValueError):
        r.score_bank(tokens, [tokens[:-1]], valid, [valid])
    with pytest.raises(ValueError):
        r.score_bank(tokens, [], valid, [])
