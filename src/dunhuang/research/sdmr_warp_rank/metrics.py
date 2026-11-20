"""R67 fixed-bank clock-error diagnostics; no artistic or routing qualification.

Errors are the driver's measured *composed original-source-clock* MAE. A score
tie receives the uniform expected error and fractional oracle success. Regret
against the current prefix and against the common full bank are both explicit.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

KS = (1, 4, 8, 16)
READOUTS = ("Single64", "Multi64", "UniformReadout64")
CONTROLS = ("actual", "zero", "swap")
TIE_TOLERANCE = 1e-12
FIELDS = ("expected_clock_error", "negative_clock_error", "oracle_regret",
          "bank_oracle_regret", "within_prefix_oracle_success",
          "within_bank_oracle_success", "prefix_oracle_error", "bank_oracle_error")


def _number(value, name, nonnegative=False):
    if value is None:
        return None
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
        raise ValueError(name + " must be finite numeric evidence or None")
    if nonnegative and value < 0:
        raise ValueError(name + " cannot be negative")
    return float(value)


def _vector(value, name, n=None, nonnegative=False):
    if not isinstance(value, (list, tuple, np.ndarray)) or np.asarray(value, dtype=object).ndim != 1:
        raise ValueError(name + " must be a one-dimensional sequence")
    if n is not None and len(value) != n:
        raise ValueError(name + " must use the complete common candidate bank")
    return [_number(v, name, nonnegative) for v in value]


def _prefixes(value, n):
    if not isinstance(value, dict) or set(map(str, value)) != set(map(str, KS)):
        raise ValueError("All four preregistered K prefixes must be retained")
    result = {}
    previous = set()
    for k in KS:
        indices = value.get(str(k), value.get(k))
        if not isinstance(indices, (list, tuple)) or any(type(i) is not int or i < 0 or i >= n for i in indices):
            raise ValueError("Prefix indices must address the same genuine candidate bank")
        if len(indices) != len(set(indices)) or len(indices) > k or not previous.issubset(indices):
            raise ValueError("Prefixes must be nested, unique, and never padded above nominal K")
        result[str(k)] = list(indices)
        previous = set(indices)
    if set(result["16"]) != set(range(n)):
        raise ValueError("K16 must expose the full registered bank without hidden candidates")
    return result


def selection_metrics(errors, indices, scores=None, *, random=False, bank_oracle=None):
    """Pure tie-aware selection arithmetic; never choose a best hyperparameter."""
    values = [errors[i] for i in indices]
    result = {"candidate_indices": list(indices), "registered_prefix_candidates": len(indices),
              "finite_error_candidates": sum(v is not None for v in values),
              "status": "NOT_EVALUABLE", "reason": None,
              "prefix_oracle_error": min(values) if values and all(v is not None for v in values) else None,
              "bank_oracle_error": bank_oracle,
              "expected_clock_error": None, "negative_clock_error": None,
              "oracle_regret": None, "bank_oracle_regret": None,
              "within_prefix_oracle_success": None, "within_bank_oracle_success": None,
              "selected_tied_indices": [], "tied_selection_count": 0,
              "max_score": None, "selection_scope": "UNIFORM_RANDOM_EXPECTATION" if random else "UNIFORM_EXPECTATION_OVER_ALL_SCORE_MAXIMA"}
    if not values:
        result["reason"] = "NO_GENUINE_CANDIDATES"
        return result
    if any(v is None for v in values):
        result["reason"] = "MISSING_COMPOSED_CLOCK_ERROR_IN_PREFIX"
        return result
    if random:
        chosen = list(indices)
        result["finite_score_candidates"] = None
    else:
        own = [scores[i] for i in indices] if scores is not None else [None] * len(indices)
        result["finite_score_candidates"] = sum(v is not None for v in own)
        if any(v is None for v in own):
            result["reason"] = "INCOMPLETE_SCORE_PREFIX_NOT_REDUCED_TO_EASY_SUBSET"
            return result
        maximum = max(own)
        chosen = [i for i, v in zip(indices, own) if abs(v - maximum) <= TIE_TOLERANCE]
        result["max_score"] = maximum
    error = float(np.mean([errors[i] for i in chosen]))
    oracle = result["prefix_oracle_error"]
    result.update(status="EVALUABLE", reason=None, expected_clock_error=error,
                  negative_clock_error=-error, oracle_regret=max(0., error - oracle),
                  bank_oracle_regret=max(0., error - bank_oracle) if bank_oracle is not None else None,
                  within_prefix_oracle_success=float(np.mean([abs(errors[i] - oracle) <= TIE_TOLERANCE for i in chosen])),
                  within_bank_oracle_success=(float(np.mean([abs(errors[i] - bank_oracle) <= TIE_TOLERANCE for i in chosen]))
                                              if bank_oracle is not None else None),
                  selected_tied_indices=chosen, tied_selection_count=len(chosen))
    return result


def _source_unit(row):
    return row.get("raw_motion_sha256") or row["source_uid"]


def _balanced(rows, field, seed, draws):
    cells = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = row.get(field)
        if value is not None:
            cells[row["group"]][row["source_unit"]].append(value)
    groups = {g: {"source_means": {s: float(np.mean(v)) for s, v in sorted(sources.items())},
                   "sources": len(sources), "queries": sum(map(len, sources.values())),
                   "mean": float(np.mean([np.mean(v) for v in sources.values()]))}
              for g, sources in sorted(cells.items())}
    values = np.asarray([v["mean"] for v in groups.values()], dtype=float)
    ci = None
    if len(values) >= 2:
        rng = np.random.default_rng(seed)
        samples = values[rng.integers(len(values), size=(draws, len(values)))].mean(1)
        ci = np.quantile(samples, [.025, .975]).tolist()
    return {"registered_queries": len(rows), "evaluable_queries": sum(v["queries"] for v in groups.values()),
            "sources": sum(v["sources"] for v in groups.values()), "music_groups": len(groups),
            "group_means": groups, "mean": float(values.mean()) if len(values) else None,
            "ci95": ci, "bootstrap_draws": draws if ci is not None else 0, "bootstrap_seed": seed,
            "balance": "QUERIES_WITHIN_RAW_RECORDING; RECORDINGS_WITHIN_MUSIC; EQUAL_MUSIC_GROUPS",
            "scope": "HISTORICALLY_EXPOSED_REGISTERED_MUSIC_GROUP_DIAGNOSTIC_NOT_NEW_GENERALIZATION",
            "six_group_interpretation": "SMALL_EXPOSED_CLUSTER_ESTIMATE_NOT_SIX_UNSEEN_INDEPENDENT_WORKS"}


def _interaction(conditions):
    result = {}
    for control in CONTROLS:
        values = {name: conditions[control][method][str(k)]["expected_clock_error"]
                  for name, method, k in (("K1Single", "Single64", 1), ("K16Single", "Single64", 16),
                                           ("K1Multi", "Multi64", 1), ("K16Multi", "Multi64", 16))}
        complete = all(v is not None for v in values.values())
        result[control] = {"four_cells_error": values, "complete_four_cells": complete,
                           "warp_gain_single_negative_error": None, "warp_gain_multi_negative_error": None,
                           "negative_clock_error_interaction": None, "combined_gain_vs_K1Single": None}
        if complete:
            a, b, c, d = (values[key] for key in ("K1Single", "K16Single", "K1Multi", "K16Multi"))
            result[control].update(warp_gain_single_negative_error=a - b,
                warp_gain_multi_negative_error=c - d,
                negative_clock_error_interaction=(c - d) - (a - b), combined_gain_vs_K1Single=a - d)
    return result


def summarize(rows, seed=67067, draws=3000):
    """All K, methods and controls survive; states are never pooled for a win."""
    if type(seed) is not int or type(draws) is not int or draws < 1:
        raise ValueError("Integer seed and positive bootstrap draws are required")
    rows = list(rows)
    seen = set(); source_groups = {}; methods = set(READOUTS)
    for row in rows:
        if any(not isinstance(row.get(key), str) or not row[key] for key in ("uid", "group", "source_uid")):
            raise ValueError("Every registered query needs a UID, recording and music-group identity")
        if row.get("state") not in ("untouched", "injected"):
            raise ValueError("Untouched and injected states must remain explicitly separate")
        identity = (row["uid"], row["state"])
        if identity in seen:
            raise ValueError("Duplicate query/state must not be counted twice")
        seen.add(identity)
        unit = _source_unit(row)
        if unit in source_groups and source_groups[unit] != row["group"]:
            raise ValueError("One raw-recording balance unit spans incompatible music bootstrap groups")
        source_groups[unit] = row["group"]
        for control, scores in row.get("scores_by_control", {}).items():
            if control not in CONTROLS or not isinstance(scores, dict):
                raise ValueError("Unknown content control or invalid score bank")
            for name in scores:
                if name not in (*READOUTS, "Energy", "RawDiagonal", "Deformation"):
                    raise ValueError("Unknown undeclared readout/baseline method: " + str(name))
                methods.add(name)
        if "costs" in row or "deformation_costs" in row:
            methods.add("Deformation")
    methods = [*READOUTS] + sorted(methods - set(READOUTS)) + ["Random"]
    queries = []
    for row in rows:
        errors = _vector(row["candidate_errors"], "candidate_errors", nonnegative=True)
        n = len(errors); prefixes = _prefixes(row["prefixes"], n)
        if row.get("actual_candidate_count", n) != n:
            raise ValueError("Actual candidate count differs from the unpadded registered bank")
        finite_errors = [v for v in errors if v is not None]
        bank_oracle = min(finite_errors) if len(finite_errors) == n and n else None
        supplied = _number(row.get("bank_oracle_min"), "bank_oracle_min", True)
        if supplied is not None and (bank_oracle is None or abs(supplied - bank_oracle) > 1e-9):
            raise ValueError("Supplied bank oracle does not equal the registered composed-clock errors")
        domain = _number(row.get("full_domain_oracle_error"), "full_domain_oracle_error", True)
        if domain is not None and bank_oracle is not None and domain > bank_oracle + 1e-9:
            raise ValueError("Full-domain oracle cannot be worse than its candidate-bank subset")
        costs = row.get("deformation_costs", row.get("costs"))
        if costs is not None:
            costs = _vector(costs, "deformation_costs", n, True)
        conditions = {}
        for control in CONTROLS:
            conditions[control] = {}
            for method in methods:
                provided = row.get("scores_by_control", {}).get(control, {}).get(method)
                if provided is not None:
                    provided = _vector(provided, "scores_by_control", n)
                elif method == "Deformation" and costs is not None:
                    provided = [None if value is None else -value for value in costs]
                by_k = {}
                for k in KS:
                    result = selection_metrics(errors, prefixes[str(k)], provided,
                                               random=method == "Random", bank_oracle=bank_oracle)
                    result.update(nominal_K=k, full_nominal_budget=len(prefixes[str(k)]) == k,
                                  candidate_budget_scope="GENUINE_UNPADDED_PREFIX; NOMINAL_K_MAY_BE_CAPPED")
                    by_k[str(k)] = result
                conditions[control][method] = by_k
        queries.append({"uid": row["uid"], "group": row["group"], "source_uid": row["source_uid"],
                        "source_unit": _source_unit(row), "state": row["state"],
                        "actual_candidate_count": n, "candidate_errors": errors, "prefixes": prefixes,
                        "bank_oracle_error": bank_oracle, "full_domain_oracle_error": domain,
                        "bank_domain_coverage_gap": (max(0., bank_oracle - domain)
                                                     if domain is not None and bank_oracle is not None else None),
                        "conditions": conditions, "interaction": _interaction(conditions),
                        "source_exposure": row.get("source_exposure", "HISTORICALLY_EXPOSED"),
                        "semantic_event_truth": "UNKNOWN"})
    states = {}
    for state in sorted({row["state"] for row in queries}):
        own = [r for r in queries if r["state"] == state]
        conditions, interactions = {}, {}
        for control in CONTROLS:
            conditions[control] = {}
            for method in methods:
                doses = {}
                for k in KS:
                    values = [{"group": row["group"], "source_unit": row["source_unit"],
                               **row["conditions"][control][method][str(k)]} for row in own]
                    doses[str(k)] = {
                        "registered_queries": len(values),
                        "status_counts": {status: sum(r["status"] == status for r in values)
                                          for status in ("EVALUABLE", "NOT_EVALUABLE")},
                        "nominal_budget_met_queries": sum(v["full_nominal_budget"] for v in values),
                        "failure_reasons": {reason: sum(v["reason"] == reason for v in values)
                                            for reason in sorted({v["reason"] for v in values if v["reason"]})},
                        "metrics": {field: _balanced(values, field, seed, draws) for field in FIELDS}}
                conditions[control][method] = doses
            interaction_rows = [{"group": row["group"], "source_unit": row["source_unit"],
                                 **row["interaction"][control]} for row in own]
            interactions[control] = {field: _balanced(interaction_rows, field, seed, draws)
                                     for field in ("warp_gain_single_negative_error", "warp_gain_multi_negative_error",
                                                   "negative_clock_error_interaction", "combined_gain_vs_K1Single")}
        states[state] = {"registered_queries": len(own), "music_groups": len({r["group"] for r in own}),
                         "recording_balance_units": len({r["source_unit"] for r in own}),
                         "conditions": conditions, "primary_2x2_interaction": interactions}
    return {
        "schema": "R67_FIXED_BANK_COMPOSED_CLOCK_METRICS_V1", "registered_queries": len(rows),
        "queries": queries, "states": states, "methods": methods, "ks": list(KS),
        "tie_tolerance": TIE_TOLERANCE, "tie_policy": "EXPECTED_ERROR_AND_FRACTIONAL_ORACLE_SUCCESS_OVER_ALL_MAXIMA",
        "error_units": "ORIGINAL_SOURCE_FRAMES_COMPOSED_PHYSICAL_CLOCK_MAE",
        "regret_scope": "oracle_regret=selected_minus_prefix_oracle; bank_oracle_regret=selected_minus_common_full_bank_oracle",
        "interaction_definition": "(Y16Multi-Y1Multi)-(Y16Single-Y1Single); Y=-CLOCK_MAE; SAME_QUERY_BEFORE_BALANCE",
        "maximum_score_is_not_a_success_metric": True,
        "Random_scope": "UNIFORM_EXPECTATION_OVER_GENUINE_PREFIX_ERRORS_NOT_A_SEEDED_DRAW",
        "missing_scores_are_not_zero_or_ignored_candidates": True,
        "operational_qualified": False, "can_route": False, "artistic_quality": "NOT_EVALUABLE",
        "semantic_event_truth": "UNKNOWN", "neural_training_steps": 0,
        "status": "EXPOSED_FIXED_BANK_CLOCK_DIAGNOSTIC" if rows else "NOT_EVALUABLE_NO_QUERIES"}


summarize_queries = summarize
