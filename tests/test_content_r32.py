import importlib
import numpy as np
import pytest

def test_core_extraction_keeps_all_observed_rotations():
    from dunhuang.research.content_r32.common import core_body,core_points
    x=np.arange(256*139).reshape(256,139)
    assert np.array_equal(core_body({"model_raw":x}),x[64:160,7:])
    assert np.array_equal(core_body({"model_raw":x[:96]}),x[:96,7:])
    points=np.zeros((256,22,3));assert core_points({"model_points":points}).shape==(96,22,3)

@pytest.mark.parametrize("name",["common","prepare","readout"])
def test_cli_imports(name):
    importlib.import_module("dunhuang.research.content_r32."+name)
