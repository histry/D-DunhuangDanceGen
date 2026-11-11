"""Convert a validated Event Plan's source->music phase map to soft SO(3) prompts.

The returned prompts are *conditions*, not generated frames or Native admissions.
Only planned joints are interpolated; Root/contact remain unconditioned.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from dunhuang.contracts import matrices, motion151, sixd
from dunhuang.io import checked


def mapped_neighborhoods(plan, frames, radius):
    """Return contiguous music-clock reference poses and cosine-squared weights.

    Integer source-frame copying would discard the time warp in Event Plan.
    Here each real music frame is mapped back to fractional source time, with
    geodesic SO(3) interpolation, without clipping or inventing observations.
    """
    if type(radius) is not int or radius < 1:
        raise ValueError('Positive integer temporal radius required')
    if plan['frames'] != frames or plan['fps'] != 30:
        raise ValueError('Phase-guidance music clock differs')
    rows, weights, occupied = [], [], set()
    for event in plan['events']:
        primitive = event['primitive']
        center, tick = primitive['source_frame'], primitive['music_frame']
        joints = primitive['joints']
        a, b = event['source_interval']
        start, end = event['music_interval']
        if tick - radius < start or tick + radius >= end or not a <= center < b:
            raise ValueError('Phase-guidance context outside planned event')
        warp = np.asarray(event['time_warp'], dtype=np.float64)
        if warp.ndim != 2 or warp.shape[1] != 2 or not np.isfinite(warp).all() or np.any(np.diff(warp, axis=0) <= 0):
            raise ValueError('Invalid phase-guidance time map')
        if warp[0].tolist() != [a, start] or warp[-1].tolist() != [b, end]:
            raise ValueError('Phase-guidance interval endpoints differ')
        target = np.arange(tick - radius, tick + radius + 1, dtype=np.int64)
        if any(int(t) in occupied for t in target):
            raise ValueError('Overlapping Event Plan prompt neighborhoods')
        occupied.update(map(int, target))
        source_position = np.interp(target, warp[:, 1], warp[:, 0])
        motion = motion151(np.load(checked(event['reference']), allow_pickle=False))
        if source_position.min() < a or source_position.max() > b - 1 or b > len(motion):
            raise ValueError('Phase-guidance source time outside observed event')
        # Other channels are ignored by anchor_windows; nearest observations are
        # used only to populate a well-formed 151D request, never to fill a target.
        poses = motion[np.rint(source_position).astype(int)].copy()
        source_grid = np.arange(a, b, dtype=np.float64)
        for joint in joints:
            channel = slice(7 + 6 * joint, 13 + 6 * joint)
            source_rotations = Rotation.from_matrix(matrices(motion[a:b, channel]))
            poses[:, channel] = sixd(Slerp(source_grid, source_rotations)(source_position).as_matrix()).astype(np.float32)
        for offset, frame, pose in zip(range(-radius, radius + 1), target, poses):
            rows.append(dict(frame=int(frame), pose=pose, joints=joints))
            weights.append(float(np.cos(np.pi * offset / (2 * radius)) ** 2) if abs(offset) < radius else 0.)
    return rows, weights
