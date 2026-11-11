"""A restricted, explicitly unqualified upper-body training probe.

Missing coordinates use neutral normalized placeholders; they are NOT a
valid 151D motion target and must not be treated as such in model evaluation.
"""
import numpy as np


def upperbody_probe_target(normalized, supported):
    x = np.asarray(normalized, dtype=np.float32)
    valid = np.asarray(supported, dtype=bool)
    if x.shape != (150, 151) or valid.shape != x.shape or not np.isfinite(x).all():
        raise ValueError('Partial probe requires finite [150,151] arrays')
    if not valid[:, 79:127].all():
        return None
    result = np.zeros_like(x)
    result[:, 79:127] = x[:, 79:127]
    return result
