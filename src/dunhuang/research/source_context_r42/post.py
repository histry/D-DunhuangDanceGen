"""Apply the same frozen R39 RTS to either network output; never refit parameters."""
from functools import lru_cache
import numpy as np
from dunhuang.research.complete_r36.core import read
from dunhuang.research.stabilize_r39.smoothing import residuals


@lru_cache(maxsize=1)
def settings():
    p='outputs/stabilize_r39_01';cal=read(p+'/calibration/result.json')['parameters'];z=np.load(p+'/calibration/arrays.npz');return cal,z['root_normalizer_scale'],z['root_normalizer_offset']


def frozen_rts(raw,anchor):
    cal,scale,offset=settings()
    return np.stack([residuals(x,a,[104,152],cal,scale,offset,.6)[0] for x,a in zip(raw,anchor)])
