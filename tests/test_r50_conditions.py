import copy
import pytest

torch = pytest.importorskip("torch")
from dunhuang.research.transition_planning_r50.conditions import MusicReperformanceCondition, diagnose


def packet():
    return MusicReperformanceCondition(
        torch.zeros(1,256,35), torch.zeros(1,128,22,9), torch.linspace(0,127,256)[None],
        torch.zeros(1,128,8), torch.zeros(1,256,8), torch.zeros(1,128,2),
        torch.ones(1,128,22,dtype=torch.bool), torch.ones(1,256,dtype=torch.bool),
        torch.zeros(1,8,139),torch.zeros(1,8,139))


def test_phase_probe_uses_missing_source_frames_and_protected_queries():
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result = diagnose()
    finally:
        torch.set_num_threads(before)
    assert result["invalid_frame_count"] == 64
    assert result["protected_frame_count"] == 16
    assert result["hidden_poison_invariant"]
    assert min(result["reference_and_query_phase_gradient_norm"]) > 0


def test_condition_contract_rejects_observed_nan_and_backward_source_clock():
    p = packet()
    p.validate()
    p.reference_features[:,64:] = float("nan")
    with pytest.raises(ValueError,match="Nonfinite valid reference"):
        p.validate()
    p.observed_mask[:,64:] = False
    p.validate()
    p.source_positions[:,100] = 0
    with pytest.raises(ValueError,match="Source time"):
        p.validate()
