"""Audit actual interpolation read support; automatic overlap is not semantic damage.

All source intervals are half-open integer frame intervals. A target sample at
q reads floor(q) and ceil(q), matching rotation_sample's interpolation support.
This module neither authorizes editing nor supplies hidden motion to a solver.
"""
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer source frame')
    return int(value)


def _spans(values, name):
    merged = []
    try:
        rows = list(values)
    except TypeError as exc:
        raise ValueError(name + ' must contain half-open intervals') from exc
    checked = []
    for row in rows:
        if (not isinstance(row, (tuple, list, np.ndarray))
                or np.ndim(row) != 1 or len(row) != 2):
            raise ValueError(name + ' must contain two-value intervals')
        a, b = (_integer(v, name) for v in row)
        if not 0 <= a < b:
            raise ValueError(name + ' requires nonempty nonnegative half-open intervals')
        checked.append((a, b))
    for a, b in sorted(checked):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def _inside(frames, spans):
    result = np.zeros(np.asarray(frames).shape, dtype=bool)
    for a, b in spans:
        result |= (frames >= a) & (frames < b)
    return result


def _evidence(lo, hi, edit, edited, spans):
    projected = _inside(lo, spans) | _inside(hi, spans)
    touched = edited[_inside(edited, spans)]
    support = sum(b - a for a, b in spans)
    return dict(
        source_spans=spans,
        target_mask=projected.tolist(),
        edit_overlap_mask=(projected & edit).tolist(),
        edit_overlap_target_frames=int(np.count_nonzero(projected & edit)),
        edited_source_frames=touched.tolist(),
        edited_source_frame_count=int(len(touched)),
        source_support_frame_count=int(support),
        source_intrusion_fraction=float(len(touched) / support) if support else None,
        evidence_status='OBSERVED_SUPPORT' if support else 'NO_EVIDENCE',
    )


def audit_mask(source_positions, edit_mask, raw_spans, band_spans, centers,
               verified_spans=None):
    """Return source/target overlap and keep unverified phase support UNKNOWN.

    Positions must be genuine, ordered source frame centers; the caller must
    separately establish that they lie within authenticated accepted support.
    ``edit_mask`` describes every target frame actually overwritten. It is
    separate from source-crop removal and is not inferred from output equality.
    """
    try:
        q = np.asarray(source_positions)
    except (TypeError, ValueError) as exc:
        raise ValueError('Finite ordered source positions required') from exc
    if (q.ndim != 1 or not len(q) or q.dtype.kind not in 'iuf'
            or not np.isfinite(q).all() or (q < 0).any()):
        raise ValueError('Finite nonnegative strictly increasing source positions required')
    q = q.astype(np.float64)
    if (np.diff(q) <= 0).any():
        raise ValueError('Strictly increasing source positions required')
    if q.max() >= np.iinfo(np.int64).max - 1:
        raise ValueError('Source positions exceed integer frame range')
    edit = np.asarray(edit_mask)
    if edit.ndim != 1 or edit.dtype.kind != 'b' or edit.shape != q.shape:
        raise ValueError('One actual boolean edit value per target frame required')
    raw = _spans(raw_spans, 'raw_spans')
    band = _spans(band_spans, 'band_spans')
    try:
        center_values = [_integer(v, 'centers') for v in centers]
    except TypeError as exc:
        raise ValueError('centers must be source-frame integers') from exc
    if any(v < 0 for v in center_values) or len(set(center_values)) != len(center_values):
        raise ValueError('centers must be unique nonnegative source frames')
    lo, hi = np.floor(q).astype(np.int64), np.ceil(q).astype(np.int64)
    edited = np.unique(np.concatenate((lo[edit], hi[edit])))
    verified = None if verified_spans is None else _spans(verified_spans, 'verified_spans')
    raw_result = _evidence(lo, hi, edit, edited, raw)
    band_result = _evidence(lo, hi, edit, edited, band)
    verified_result = None if verified is None else _evidence(lo, hi, edit, edited, verified)
    return dict(
        schema='SOURCE_TARGET_EDIT_SUPPORT_AUDIT_V1',
        scope='INTERPOLATION_READ_SUPPORT_NOT_SEMANTIC_DAMAGE_OR_PHYSICS',
        target_frames=int(len(q)),
        source_positions=q.tolist(),
        source_read_support=dict(floor=lo.tolist(), ceil=hi.tolist()),
        edit_mask=edit.tolist(),
        retained_target_mask=(~edit).tolist(),
        edit_target_frames=int(np.count_nonzero(edit)),
        edited_source_read_frames=edited.tolist(),
        edited_centers=[v for v in center_values if v in set(edited.tolist())],
        raw=raw_result,
        band=band_result,
        verified=verified_result,
        verified_status='UNKNOWN' if verified is None else 'SUPPLIED_SUPPORT_NOT_INDEPENDENTLY_AUTHENTICATED_BY_THIS_FUNCTION',
        verified_intrusion_fraction=None if verified_result is None else verified_result['source_intrusion_fraction'],
        task_status='RECONSTRUCTION' if edit.any() else 'READ_ONLY_SOURCE',
        preserved_source_claim_authorized=False,
        crop_removal_scope='NOT_COMPUTED_HERE; AUDIT_SEPARATELY_FROM_ACTUAL_RECONSTRUCTION_EDIT',
    )
