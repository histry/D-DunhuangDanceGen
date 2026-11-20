"""Frozen R41 token readouts with a shared physical-time support contract.

This module neither encodes features nor chooses a Warp. UniformReadout64 is
the deliberately redundant 32/16/16 *nonoverlapping* bin control; Uniform Warp
is a different intervention implemented by the experiment driver.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

EPS = 1e-8
METHODS = {
    "Single64": (("Single64", 1.0),),
    "Multi64": (("Fine32", .5), ("Mid16", .25), ("Coarse16", .25)),
    "UniformReadout64": (("Fine32", .5), ("Mid16", .25), ("Mid16", .25)),
}


def _hash(array, dtype):
    value = np.ascontiguousarray(array, dtype=dtype)
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\n" + value.tobytes()).hexdigest()


def _mask(value, n, name):
    value = np.asarray(value)
    if value.shape != (n,) or value.dtype != np.dtype(bool):
        raise ValueError(name + " must be a one-dimensional bool mask on the original clock")
    return value.copy()


def _tokens(value, shape=None):
    value = np.asarray(value)
    if (value.ndim != 2 or min(value.shape) < 1 or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all() or (shape is not None and value.shape != shape)):
        raise ValueError("Finite token matrices must share exactly the same [time, feature] shape")
    return value.astype(np.float64, copy=False)


def _scale(centres, support, count, width_multiplier):
    # The domain is fixed by the entire original clock, not compressed valid indices.
    left = centres[0] - (centres[1] - centres[0]) / 2
    right = centres[-1] + (centres[-1] - centres[-2]) / 2
    width = (right - left) / count
    middle = left + (np.arange(count) + .5) * width
    lo = np.maximum(left, middle - width * width_multiplier / 2)
    hi = np.minimum(right, middle + width * width_multiplier / 2)
    membership = ((centres[None] >= lo[:, None]) & (centres[None] < hi[:, None]))
    membership &= support[None]
    coverage = membership.sum(0)
    if np.any(support & (coverage == 0)):
        raise AssertionError("Readout cells left a supported frame uncovered")
    weights = membership / np.maximum(coverage, 1)[None]
    mass = weights.sum(1)
    valid_total = int(support.sum())
    score_weight = mass / valid_total if valid_total else np.zeros(count)
    evidence = {
        "cell_count": count, "window_width_multiplier": width_multiplier,
        "original_clock_domain": [float(left), float(right)],
        "cell_intervals_half_open": np.c_[lo, hi].tolist(),
        "bin_valid_counts": membership.sum(1).tolist(),
        "empty_bin_indices": np.flatnonzero(mass == 0).tolist(),
        "cell_mass": mass.tolist(), "cell_score_weight": score_weight.tolist(),
        "per_token_window_coverage": coverage.tolist(),
        "per_token_total_pool_weight": weights.sum(0).tolist(),
        "sparse_pool_weights": [
            {"indices": np.flatnonzero(row).tolist(), "weights": row[row > 0].tolist()}
            for row in weights],
        "pool_weights_sha256": _hash(weights, "<f8"),
        "weight_rule": "MEMBERSHIP_DIVIDED_BY_OVERLAP_COUNT; CELL_SCORE_WEIGHT=MASS/COMMON_VALID_COUNT",
        "no_time_hole_compression": True,
    }
    return weights, mass, score_weight, evidence


def _pool(tokens, weights, mass):
    pooled = weights @ tokens / np.maximum(mass, 1e-300)[:, None]
    norms = np.linalg.norm(pooled, axis=1)
    return pooled / np.maximum(norms, EPS)[:, None], norms


def score_bank(music_tokens, motion_tokens_list, music_valid, motion_valid_list,
               centres=None, fixed_support=None):
    """Return all candidate scores without selecting a candidate or fitting weights.

    A supplied fixed_support is preserved exactly or rejected if unsupported by
    any input. This lets actual/zero/swap calls use one predeclared support.
    Empty bins or pooled norms <= EPS make the affected method/candidate None;
    missing evidence is never replaced by a numeric zero score.
    """
    music = _tokens(music_tokens)
    motions = [_tokens(value, music.shape) for value in motion_tokens_list]
    n = len(music)
    if n < 2 or not motions or len(motion_valid_list) != len(motions):
        raise ValueError("At least two clock centres and a nonempty aligned motion bank are required")
    mv = _mask(music_valid, n, "music_valid")
    dvs = [_mask(value, n, "motion_valid") for value in motion_valid_list]
    if centres is None:
        centres = np.arange(n, dtype=float)
    else:
        centres = np.asarray(centres)
        if (centres.shape != (n,) or centres.dtype.kind not in "fiu"
                or not np.isfinite(centres).all() or not np.all(np.diff(centres) > 0)):
            raise ValueError("Actual centres must be a finite strictly increasing one-dimensional clock")
        centres = centres.astype(np.float64, copy=False)
    common = mv.copy()
    for valid in dvs:
        common &= valid
    if fixed_support is not None:
        supplied = _mask(fixed_support, n, "fixed_support")
        if np.any(supplied & ~common):
            raise ValueError("fixed_support contains frames not observed by this complete bank; no silent intersection")
        common = supplied
    scales = {}
    prepared = {}
    for name, count, width in (("Single64", 64, 1), ("Fine32", 32, 1),
                               ("Mid16", 16, 1), ("Coarse16", 16, 2)):
        weights, mass, score_weight, evidence = _scale(centres, common, count, width)
        pooled_music, music_norms = _pool(music, weights, mass)
        prepared[name] = (weights, mass, score_weight, pooled_music, music_norms)
        scales[name] = evidence
    scores, statuses = {}, {}
    for method, components in METHODS.items():
        scores[method], statuses[method] = [], []
        for index, motion in enumerate(motions):
            reasons, components_result = [], []
            combined = 0.0
            for name, scale_weight in components:
                weights, mass, cell_weights, pooled_music, music_norms = prepared[name]
                pooled_motion, motion_norms = _pool(motion, weights, mass)
                empty = np.flatnonzero(mass == 0).tolist()
                near_music = np.flatnonzero((mass > 0) & (music_norms <= EPS)).tolist()
                near_motion = np.flatnonzero((mass > 0) & (motion_norms <= EPS)).tolist()
                if empty:
                    reasons.append(name + ":EMPTY_BIN")
                if near_music:
                    reasons.append(name + ":MUSIC_POOLED_NORM_TOO_SMALL")
                if near_motion:
                    reasons.append(name + ":MOTION_POOLED_NORM_TOO_SMALL")
                score = (None if empty or near_music or near_motion else
                         float(np.clip(np.dot(cell_weights, np.sum(pooled_music * pooled_motion, axis=1)), -1, 1)))
                components_result.append({"scale": name, "weight": scale_weight, "score": score,
                                          "empty_bins": empty, "near_zero_music_bins": near_music,
                                          "near_zero_motion_bins": near_motion})
                if score is not None:
                    combined += scale_weight * score
            failed = bool(reasons)
            scores[method].append(None if failed else float(np.clip(combined, -1, 1)))
            statuses[method].append({"candidate_index": index,
                                      "status": "NOT_EVALUABLE" if failed else "SCORED",
                                      "reasons": sorted(set(reasons)), "components": components_result})
    return {
        "schema": "R67_FROZEN_COMMON_SUPPORT_READOUT_V1", "scores": scores,
        "candidate_status": statuses, "scales": scales,
        "method_components": {key: [{"scale": name, "weight": value} for name, value in components]
                              for key, components in METHODS.items()},
        "support": {"input_centres": n, "candidate_count": len(motions),
                    "common_valid_count": int(common.sum()), "common_valid_fraction": float(common.mean()),
                    "common_valid_mask": common.tolist(), "centres_sha256": _hash(centres, "<f8"),
                    "common_support_sha256": _hash(common, "|b1"),
                    "music_valid_sha256": _hash(mv, "|b1"),
                    "motion_valid_sha256": [_hash(value, "|b1") for value in dvs],
                    "music_tokens_sha256": _hash(music, "<f8"),
                    "motion_tokens_sha256": [_hash(value, "<f8") for value in motions],
                    "rule": "SUPPLIED_FIXED_SUPPORT_UNCHANGED" if fixed_support is not None else "MAXIMUM_BANK_MASK_INTERSECTION",
                    "mask_holes_compressed": False},
        "pool_normalization": "WEIGHTED_MEAN_THEN_L2; NORM_LE_EPS_IS_MISSING_NOT_ZERO_SCORE",
        "norm_epsilon": EPS, "trained_parameters": 0, "new_training_steps": 0,
        "UniformReadout64_scope": "32/16/16_NONOVERLAPPING_FULL_CLOCK_BINS; MID_AND_COARSE_INTENTIONALLY_IDENTICAL; NOT_UNIFORM_WARP",
    }
