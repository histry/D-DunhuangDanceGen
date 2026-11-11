import numpy as np
import pytest
from dunhuang.generation.probe import upperbody_probe_target


def test_partial_probe_fills_only_supported_upperbody():
    x = np.ones((150, 151), np.float32) * 4
    valid = np.zeros_like(x, bool)
    valid[:, 79:127] = True
    result = upperbody_probe_target(x, valid)
    assert np.all(result[:, 79:127] == 4)
    assert not result[:, :79].any() and not result[:, 127:].any()
    valid[42, 80] = False
    assert upperbody_probe_target(x, valid) is None


def test_partial_probe_contract_rejects_bad_shape():
    with pytest.raises(ValueError):
        upperbody_probe_target(np.ones((150, 48)), np.ones((150, 48), bool))
