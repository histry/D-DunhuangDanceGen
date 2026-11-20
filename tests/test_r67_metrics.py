"""TEST_FIXTURE metrics only: no source/semantic/human labels are fabricated."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

from dunhuang.research.sdmr_warp_rank import metrics as m


def fixture(uid="TEST_FIXTURE_0", group="TEST_FIXTURE_M0", source="TEST_FIXTURE_S0"):
    n = 16
    row = {"uid": uid, "group": group, "source_uid": source, "state": "injected",
           "candidate_errors": list(map(float, range(n))),
           "prefixes": {str(k): list(range(k)) for k in m.KS}, "actual_candidate_count": n,
           "bank_oracle_min": 0., "full_domain_oracle_error": 0., "scores_by_control": {}}
    for control in m.CONTROLS:
        row["scores_by_control"][control] = {method: [-float(i) for i in range(n)] for method in m.READOUTS}
    return row


def test_ties_use_expected_error_and_fractional_oracle_success():
    result = m.selection_metrics([0., 4., 9.], [0, 1, 2], [1., 1., 0.], bank_oracle=0.)
    assert result["expected_clock_error"] == 2
    assert result["within_bank_oracle_success"] == .5
    assert result["oracle_regret"] == 2 and result["tied_selection_count"] == 2


def test_random_is_exact_full_prefix_expectation():
    result = m.selection_metrics([2., 4., 6.], [0, 1, 2], random=True, bank_oracle=2.)
    assert result["expected_clock_error"] == 4
    assert result["within_prefix_oracle_success"] == pytest.approx(1/3)
    assert result["oracle_regret"] == 2


def test_missing_candidate_does_not_shrink_to_easy_subset():
    row = fixture(); row["scores_by_control"]["actual"]["Multi64"][15] = None
    summary = m.summarize([row], draws=8)
    q = summary["queries"][0]
    assert q["conditions"]["actual"]["Multi64"]["16"]["expected_clock_error"] is None
    assert q["conditions"]["actual"]["Multi64"]["8"]["expected_clock_error"] == 0
    assert q["interaction"]["actual"]["negative_clock_error_interaction"] is None
    assert summary["registered_queries"] == 1


def test_interaction_is_same_query_difference_of_negative_error_gains():
    row = fixture()
    row["candidate_errors"] = [8., 7., 6., 5., 4., 3., 2., 1., 0., 9., 10., 11., 12., 13., 14., 15.]
    row["scores_by_control"]["actual"]["Single64"] = [0.] * 16
    row["scores_by_control"]["actual"]["Single64"][4] = 1.
    row["scores_by_control"]["actual"]["Multi64"] = [0.] * 16
    row["scores_by_control"]["actual"]["Multi64"][8] = 1.
    q = m.summarize([row], draws=8)["queries"][0]
    assert q["interaction"]["actual"]["warp_gain_single_negative_error"] == 4.
    assert q["interaction"]["actual"]["warp_gain_multi_negative_error"] == 8.
    assert q["interaction"]["actual"]["negative_clock_error_interaction"] == 4.


def test_prefix_oracle_is_distinct_from_common_bank_oracle():
    result = m.selection_metrics([4., 0.], [0], [1., 0.], bank_oracle=0.)
    assert result["within_prefix_oracle_success"] == 1
    assert result["within_bank_oracle_success"] == 0
    assert result["oracle_regret"] == 0 and result["bank_oracle_regret"] == 4


def test_recordings_then_music_balance_not_query_iid():
    rows = []
    for i, (g, s, error) in enumerate([("G0", "S0", 0.), ("G0", "S0", 0.),
                                       ("G0", "S1", 4.), ("G1", "S2", 10.)]):
        row = fixture(f"TEST_FIXTURE_{i}", g, s)
        row["candidate_errors"] = [error] * 16; row["bank_oracle_min"] = error
        row["full_domain_oracle_error"] = None
        rows.append(row)
    metric = m.summarize(rows, draws=20)["states"]["injected"]["conditions"]["actual"]["Single64"]["1"]["metrics"]["expected_clock_error"]
    assert metric["mean"] == 6.  # mean(mean(S0=0,S1=4)=2, S2=10)
    assert metric["music_groups"] == 2 and metric["sources"] == 3
    assert metric["evaluable_queries"] == 4 and metric["bootstrap_draws"] == 20


def test_states_remain_separate_and_fewer_than_two_groups_have_no_ci():
    row = fixture(); other = copy.deepcopy(row); other["state"] = "untouched"
    result = m.summarize([row, other], draws=8)
    assert set(result["states"]) == {"untouched", "injected"}
    metric = result["states"]["untouched"]["conditions"]["actual"]["Multi64"]["16"]["metrics"]["expected_clock_error"]
    assert metric["ci95"] is None and result["artistic_quality"] == "NOT_EVALUABLE"


def test_baselines_and_capped_bank_are_retained_without_padding():
    row = fixture(); row["candidate_errors"] = [0., 3.]; row["actual_candidate_count"] = 2
    row["prefixes"] = {"1": [0], "4": [0, 1], "8": [0, 1], "16": [0, 1]}
    row["costs"] = [4., 1.]
    for controls in row["scores_by_control"].values():
        for method in controls:
            controls[method] = [1., 0.]
        controls["Energy"] = [0., 1.]; controls["RawDiagonal"] = [0., 0.]
    result = m.summarize([row], draws=8)
    by = result["queries"][0]["conditions"]["actual"]
    assert by["Deformation"]["16"]["expected_clock_error"] == 3.
    assert by["Random"]["16"]["expected_clock_error"] == 1.5
    assert by["Energy"]["16"]["full_nominal_budget"] is False


def test_invalid_input_oracle_prefix_and_leakage_are_rejected():
    row = fixture()
    for mutate in (lambda x: x.update(bank_oracle_min=2.),
                   lambda x: x["prefixes"].update({"16": [0, 1]}),
                   lambda x: x["candidate_errors"].__setitem__(0, True)):
        bad = copy.deepcopy(row); mutate(bad)
        with pytest.raises(ValueError): m.summarize([bad], draws=8)
    other = fixture("TEST_FIXTURE_OTHER", "OTHER_GROUP", row["source_uid"])
    with pytest.raises(ValueError, match="bootstrap groups"): m.summarize([row, other], draws=8)
    with pytest.raises(ValueError, match="Duplicate"): m.summarize([row, row], draws=8)


def test_empty_population_and_no_evidence_are_not_zero_success():
    result = m.summarize([], draws=8)
    assert result["status"] == "NOT_EVALUABLE_NO_QUERIES" and result["states"] == {}
    row = fixture(); row["candidate_errors"] = [None] * 16; row["bank_oracle_min"] = None
    q = m.summarize([row], draws=8)["queries"][0]
    assert q["conditions"]["actual"]["Random"]["16"]["expected_clock_error"] is None
    assert q["conditions"]["actual"]["Random"]["16"]["reason"] == "MISSING_COMPOSED_CLOCK_ERROR_IN_PREFIX"
