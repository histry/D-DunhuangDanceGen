"""One explicit world-to-window transform, with exact protected-context restoration."""
import numpy as np
from dunhuang.contracts import MotionAdapter
def origin(native):
    value=native[0,4:7].copy();value[1]=0.;return value
def encode(native):
    o=origin(native);return MotionAdapter.to_edge(native,o),o
def restore(native,o):
    x=native.copy();x[:,4:7]+=o;return x
