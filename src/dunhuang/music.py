"""Pure timing/contact adaptation for an isolated continuous EDGE baseline.

No frozen Core/Bridge assembly or relative repair permission is inherited.
"""
import numpy as np


def clock_plan(samples, sample_rate):
    if type(samples) is not int or type(sample_rate) is not int or samples <= 0 or sample_rate <= 0:
        raise ValueError('Actual positive audio sample count and rate required')
    frames = (samples * 30 + sample_rate - 1) // sample_rate
    windows = max(1, (max(0, frames - 150) + 74) // 75 + 1)
    covered = 150 + (windows - 1) * 75
    padded_samples = (covered * sample_rate + 29) // 30
    return dict(fps=30, frames=frames, horizon=150, hop=75, windows=windows,
                covered_frames=covered, padding_frames=covered-frames,
                original_samples=samples, sample_rate=sample_rate,
                padded_samples=padded_samples, silence_samples=padded_samples-samples,
                starts=[75*i for i in range(windows)])


def blend_contacts(windows):
    x=np.asarray(windows)
    if x.ndim!=3 or x.shape[1:]!=(150,4) or len(x)<1 or not np.isfinite(x).all():
        raise ValueError('Actual finite model contact windows required')
    out=np.zeros((150+75*(len(x)-1),4),np.float32)
    # Same temporal fade as official long-mode Root stitching. These are model
    # predictions; Native derives its own contact population from actual FK.
    for i,row in enumerate(x):
        w=np.ones(150,np.float32)
        if i: w[:75]=np.linspace(0,1,75,dtype=np.float32)
        if i<len(x)-1: w[75:]=np.linspace(1,0,75,dtype=np.float32)
        out[i*75:i*75+150]+=row.astype(np.float32)*w[:,None]
    return out

