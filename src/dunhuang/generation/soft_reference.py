"""Soft source-backed event prompts for the frozen EDGE denoiser.

They are conditions, never pasted output poses. Unsupported joints are masked,
not clipped. Native remains the only full physical acceptance authority.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from dunhuang.contracts import MotionAdapter, matrices, sixd, motion151
from dunhuang.io import checked

SCHEMA = 'soft_event_reference_v1'
JOINTS = tuple(range(12, 20))


def build_soft_reference(plan, frames, scale, bias):
    """Return normalized [W,150,151], mask [W,150,151], phase [W,150,1], audit.

    Windows share exactly the same values in their overlapping 75 frames.
    Only supported source joints 12..19 are observed; no output target is read.
    """
    scale = np.asarray(scale, dtype=np.float32).reshape(-1)
    bias = np.asarray(bias, dtype=np.float32).reshape(-1)
    if (scale.shape != (151,) or bias.shape != (151,) or
            not np.isfinite(scale).all() or not np.isfinite(bias).all() or
            np.any(scale == 0)):
        raise ValueError('Frozen Normalizer scale/bias contract differs')
    if type(frames) is not int or frames < 1 or plan.get('frames') != frames:
        raise ValueError('Event prompt clock mismatch')
    windows = max(1, (max(0, frames - 150) + 74) // 75 + 1)
    covered = 150 + 75 * (windows - 1)
    values = np.zeros((covered, 151), np.float32)
    masks = np.zeros((covered, 151), bool)
    phase = np.zeros((covered, 1), np.float32)
    total = supported = 0
    cursor = 0
    for event in plan['events']:
        a, b = event['music_interval']
        lo, hi = event['source_interval']
        if a != cursor or not a < b <= frames:
            raise ValueError('Event prompt requires contiguous music intervals')
        source = motion151(np.asarray(np.load(checked(event['reference']), allow_pickle=False)))
        if not 0 <= lo < hi <= len(source):
            raise ValueError('Event prompt source interval invalid')
        warp = np.asarray(event['time_warp'], np.float64)
        if (warp.ndim != 2 or warp.shape[1] != 2 or
                not np.isfinite(warp).all() or np.any(np.diff(warp, axis=0) <= 0) or
                not np.array_equal(warp[0], [lo, a]) or
                not np.array_equal(warp[-1], [hi, b])):
            raise ValueError('Event prompt time map differs')
        times = np.arange(a, b)
        positions = np.interp(times, warp[:, 1], warp[:, 0])
        positions = np.clip(positions, lo, hi - 1)
        left = np.floor(positions).astype(int)
        right = np.minimum(left + 1, hi - 1)
        alpha = (positions - left).astype(np.float32)
        first = MotionAdapter.to_edge(source[left].copy())[:, 7:].reshape(-1, 24, 6)
        second = MotionAdapter.to_edge(source[right].copy())[:, 7:].reshape(-1, 24, 6)
        phase[a:b, 0] = ((positions - lo) / (hi - lo)).astype(np.float32)
        for joint in JOINTS:
            r0 = Rotation.from_matrix(matrices(first[:, joint], row=True))
            r1 = Rotation.from_matrix(matrices(second[:, joint], row=True))
            delta = (r0.inv() * r1).as_rotvec()
            pose = sixd((r0 * Rotation.from_rotvec(delta * alpha[:, None])).as_matrix(), row=True).astype(np.float32)
            channels = slice(7 + 6 * joint, 13 + 6 * joint)
            normalized = pose * scale[channels] + bias[channels]
            good = np.isfinite(normalized).all(axis=1) & (np.abs(normalized) <= 1 + 32 * np.finfo(np.float32).eps).all(axis=1)
            values[a:b, channels] = np.where(good[:, None], normalized, 0.)
            masks[a:b, channels] = good[:, None]
            total += len(good)
            supported += int(good.sum())
        cursor = b
    if cursor != frames:
        raise ValueError('Event prompt did not cover the song')
    reference = np.stack([values[75*i:75*i+150] for i in range(windows)])
    known = np.stack([masks[75*i:75*i+150] for i in range(windows)])
    phases = np.stack([phase[75*i:75*i+150] for i in range(windows)])
    return reference, known, phases, dict(
        schema=SCHEMA, source_joint_frames=total, supported_joint_frames=supported,
        masked_unsupported_joint_frames=total-supported,
        supported_fraction=float(supported/max(total, 1)),
        full_151d_target_supported=False, physical_qualified=False,
        source_only=True, no_source_pose_paste=True)
