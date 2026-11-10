"""Metric SMPL24: contact4 + root3 + local rotations24x6, at 30 FPS.
Native uses Y-up/columns; official EDGE uses Z-up/PyTorch3D rows.
Contacts are preserved as predictions, never substituted for measured support.
"""
import numpy as np
Q = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], np.float32)
FPS, HORIZON, HOP = 30, 150, 75

def motion151(x):
    x = np.asarray(x)
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 151 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite nonempty FP32 [T,151] required')
    return x

def matrices(six, row=False):
    a, b = np.asarray(six)[..., :3], np.asarray(six)[..., 3:]
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(n < 1e-6):
        raise ValueError('Degenerate first rotation vector')
    a = a / n
    b = b - np.sum(a*b, axis=-1, keepdims=True)*a
    n = np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(n < 1e-6):
        raise ValueError('Degenerate second rotation vector')
    b = b/n
    return np.stack([a, b, np.cross(a,b)], axis=-2 if row else -1)

def sixd(m, row=False):
    return np.concatenate([m[...,0,:], m[...,1,:]], -1) if row else np.concatenate([m[..., :,0], m[..., :,1]], -1)

class MotionAdapter:
    @staticmethod
    def to_edge(x, origin=None):
        x = motion151(x); origin = np.zeros(3) if origin is None else np.asarray(origin)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError('Finite three-dimensional origin required')
        r = matrices(x[:,7:].reshape(-1,24,6)).astype(np.float64)
        r[:,0] = Q @ r[:,0]
        out = x.copy(); out[:,4:7] = (x[:,4:7]-origin) @ Q.T
        out[:,7:] = sixd(r, row=True).reshape(-1,144)
        return out

    @staticmethod
    def to_native(x, origin=None):
        x = motion151(x); origin = np.zeros(3) if origin is None else np.asarray(origin)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError('Finite three-dimensional origin required')
        r = matrices(x[:,7:].reshape(-1,24,6), row=True).astype(np.float64)
        r[:,0] = Q.T @ r[:,0]
        out = x.copy(); out[:,4:7] = x[:,4:7] @ Q + origin
        out[:,7:] = sixd(r).reshape(-1,144)
        return out
