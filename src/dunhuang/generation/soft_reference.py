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
        if (warp.ndim != 2 or warp.shape[1] != 2 or len(warp) < 2 or
                not np.isfinite(warp).all() or np.any(np.diff(warp, axis=0) <= 0) or
                not np.array_equal(warp[0], [lo, a]) or
                not np.array_equal(warp[-1], [hi, b])):
            raise ValueError('Event prompt time map differs')
        rates = np.diff(warp[:, 1]) / np.diff(warp[:, 0])
        if np.any(rates < .75 - 1e-6) or np.any(rates > 1.33 + 1e-6):
            raise ValueError('Event prompt time-map rate outside registered limits')
        times = np.arange(a, b)
        positions = np.interp(times, warp[:, 1], warp[:, 0])
        positions = np.clip(positions, lo, hi - 1)
        sampled = resample_native(source, positions)
        edge = MotionAdapter.to_edge(sampled)
        phase[a:b, 0] = ((positions - lo) / (hi - lo)).astype(np.float32)
        for joint in JOINTS:
            channels = slice(7 + 6 * joint, 13 + 6 * joint)
            normalized = edge[:, channels] * scale[channels] + bias[channels]
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
        schema=SCHEMA, source_frames=frames,
        supported_time_frames=int(masks[:frames].any(axis=1).sum()),
        source_joint_frames=total, supported_joint_frames=supported,
        masked_unsupported_joint_frames=total-supported,
        supported_fraction=float(supported/max(total, 1)),
        full_151d_target_supported=False, physical_qualified=False,
        source_only=True, no_source_pose_paste=True,
        endpoint_policy='half-open source interval: hold last observed pose for final fractional sample')


def resample_native(source, positions):
    """SO(3) geodesics for rotations, linear root, nearest predicted contacts.

    Positions must refer to observed frames. This is a source condition/target
    transform; it never reads generated output or certifies physical quality.
    """
    source = motion151(np.asarray(source))
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 1 or not len(positions) or not np.isfinite(positions).all() or positions.min()<0 or positions.max()>len(source)-1:
        raise ValueError('Source sampling outside observed motion')
    left=np.floor(positions).astype(int);right=np.minimum(left+1,len(source)-1)
    alpha=(positions-left).astype(np.float32)
    result=source[np.rint(positions).astype(int)].copy()
    result[:,4:7]=source[left,4:7]*(1-alpha[:,None])+source[right,4:7]*alpha[:,None]
    first=Rotation.from_matrix(matrices(source[left,7:].reshape(-1,24,6)).reshape(-1,3,3))
    second=Rotation.from_matrix(matrices(source[right,7:].reshape(-1,24,6)).reshape(-1,3,3))
    delta=(first.inv()*second).as_rotvec()
    rotation=first*Rotation.from_rotvec(delta*np.repeat(alpha,24)[:,None])
    result[:,7:]=sixd(rotation.as_matrix()).reshape(-1,144).astype(np.float32)
    return result
