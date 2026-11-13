import importlib.util
from pathlib import Path
import numpy as np

def test_matched_condition_norm_uses_scale_aware_fp32_rounding_bound():
    path=Path(__file__).resolve().parents[1]/"scripts/check_protected_source.py"
    spec=importlib.util.spec_from_file_location("protected_output_checks",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reference=15.392820591716085;other=15.392822332698666
    assert abs(reference-other)>1e-6
    assert module.norm_relative_error(reference,other)<2*np.finfo(np.float32).eps
    for scale in (.001,1.,1000.):
        assert module.norm_relative_error(reference*scale,other*scale)<2*np.finfo(np.float32).eps
        assert module.norm_relative_error(reference*scale,reference*scale*1.0001)>2*np.finfo(np.float32).eps
    assert module.norm_relative_error(0.,0.)==0
