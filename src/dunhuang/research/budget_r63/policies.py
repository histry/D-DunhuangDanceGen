"""Preregistered fixed-budget duration decisions, with a DEV-only 1NN rule.

Shortest-Screened may screen newly recompiled G1 candidates. The boundary rule
cannot: it receives only baseline48 observed edge features and the serialized
development fit. A rejected predicted candidate is a failure, never a fallback.
"""
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation


DURATIONS = (24, 36, 48, 60, 72)
NAMES = ('Fixed48', 'Fixed24', 'ShortestScreened', 'BoundaryConditioned')
FEATURE_NAMES = ('endpoint_pose_mean_angle_rad', 'left_omega_rms_radps',
                 'right_omega_rms_radps', 'left_alpha_rms_radps2',
                 'right_alpha_rms_radps2')


def passes(record):
    """A candidate must both construct and pass the unchanged G1 screen."""
    g1 = record.get('methods', {}).get('G1', {})
    return bool(record.get('status') == 'GEOMETRY_EVALUATED'
                and g1.get('constructible', False)
                and g1.get('passes_original_screen', False))


def shortest_screened(records):
    """First passing registered duration; absent/failed candidates are not wins."""
    by_gap = {}
    for row in records:
        gap = row.get('gap_frames')
        if (not isinstance(gap, (int, np.integer)) or isinstance(gap, (bool, np.bool_))
                or gap not in DURATIONS):
            raise ValueError('Only registered duration candidates may be screened')
        if gap in by_gap:
            raise ValueError('Duplicate duration candidate')
        by_gap[gap] = row
    for g in DURATIONS:
        # An unexecuted shorter candidate is UNKNOWN, not an observed failure.
        if g not in by_gap:
            return None
        if passes(by_gap[g]):
            return int(g)
    return None


def boundary_features(left, right):
    """Five baseline48 features from the same visible four-pose edge readset.

    The finite differences are computed in each actual endpoint tangent chart,
    with the same one-sided derivatives used by the frozen geometry. Unknown
    bridge poses, music identity and candidate quality are never input features.
    """
    L, R = np.asarray(left, float), np.asarray(right, float)
    if (L.ndim != 4 or R.ndim != 4 or L.shape[1:] != (22, 3, 3)
            or R.shape[1:] != (22, 3, 3) or min(len(L), len(R)) < 4):
        raise ValueError('Two genuine body22 boundary contexts with four poses required')
    L, R = L[-4:], R[:4]
    if not np.isfinite(L).all() or not np.isfinite(R).all():
        raise ValueError('Finite actual boundary observations required')
    delta = L[-1].swapaxes(-1, -2)@R[0]
    pose = Rotation.from_matrix(delta).magnitude().mean()
    values = []
    for block, at_end in ((L, True), (R, False)):
        center = block[-1] if at_end else block[0]
        tangent = Rotation.from_matrix(
            (center[None].swapaxes(-1, -2)@block).reshape(-1, 3, 3)
        ).as_rotvec().reshape(4, 22, 3)
        if at_end:
            omega = (3*tangent[-1]-4*tangent[-2]+tangent[-3])*15
            alpha = (tangent[-1]-2*tangent[-2]+tangent[-3])*900
        else:
            omega = (-3*tangent[0]+4*tangent[1]-tangent[2])*15
            alpha = (tangent[0]-2*tangent[1]+tangent[2])*900
        values.append((float(np.sqrt(np.mean(omega**2))),
                       float(np.sqrt(np.mean(alpha**2)))))
    return [float(pose), values[0][0], values[1][0], values[0][1], values[1][1]]


def feature_row(case, baseline, arrays):
    """Build rule input only from this case's newly recompiled baseline48."""
    if baseline['case_key'] != case['key'] or baseline['gap_frames'] != 48:
        raise ValueError('Rule features require the same case at fixed48')
    key = baseline['key']
    left, right = arrays.get(key+'_left'), arrays.get(key+'_right')
    features = boundary_features(left, right) if left is not None and right is not None else None
    return dict(key=case['key'], role=case['role'], features=features,
                left_source_uid=case['left']['source_uid'] if case['left'] else None,
                right_source_uid=case['right']['source_uid'] if case['right'] else None,
                feature_source='RECOMPILED_FIXED48_VISIBLE_EDGE_POSES_ONLY',
                baseline_key=key, baseline_status=baseline['status'])


def _features(value):
    if value is None:
        return None
    x = np.asarray(value, dtype=float)
    if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
        return None
    return x


def fit(dev_rows):
    """Fit deterministic 1NN to newly screened, fixed-budget DEV labels only.

    A missing shortest label remains excluded rather than being relabelled72 or
    a success. Normalization uses only the finite, labelled DEV rows actually
    fitted. A model with no usable rows is explicitly NOT_FIT.
    """
    rows = sorted(list(dev_rows), key=lambda r: r['key'])
    if any(r.get('role') != 'DEV' for r in rows):
        raise ValueError('Duration fit accepts DEV rows only; no EVAL leakage')
    keys = [r['key'] for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Development case keys must be unique')
    fitted, excluded = [], []
    for row in rows:
        label = row.get('shortest_gap')
        x = _features(row.get('features'))
        if label is None:
            excluded.append(dict(key=row['key'], reason='NO_PASSING_FIXED_BUDGET_G1_LABEL'))
        elif (not isinstance(label, (int, np.integer)) or isinstance(label, (bool, np.bool_))
              or label not in DURATIONS):
            raise ValueError('DEV targets must be a registered shortest gap or None')
        elif x is None:
            excluded.append(dict(key=row['key'], reason='MISSING_OR_NONFINITE_BASELINE48_FEATURES'))
        else:
            fitted.append(dict(key=row['key'], features=x.tolist(), shortest_gap=int(label),
                               left_source_uid=row.get('left_source_uid'),
                               right_source_uid=row.get('right_source_uid')))
    if fitted:
        X = np.asarray([r['features'] for r in fitted])
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std < 1e-12] = 1.
        Z = (X-mean)/std
    else:
        mean, std, Z = np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)), np.empty((0, len(FEATURE_NAMES)))
    model = dict(kind='DEV_ONLY_STANDARDIZED_1NN_V1', status='FIT' if fitted else 'NOT_FIT',
        rule='ONE_NEAREST_DEV_CASE_EUCLIDEAN; EXACT_DISTANCE_TIE_BY_LEXICOGRAPHIC_CASE_KEY',
        feature_names=list(FEATURE_NAMES), feature_source='RECOMPILED_FIXED48_VISIBLE_EDGE_POSES_ONLY',
        label_source='R63_RECOMPILED_FIXED_TOTAL_BUDGET_FIRST_PASSING_G1; NOT_R62_ORACLE_LABELS',
        fit_role='DEV', source_disjoint_evaluation_required=True,
        input_development_cases=len(rows), fitted_cases=len(fitted), excluded_cases=excluded,
        development_case_keys=keys, fitted_rows=fitted,
        normalization_mean=mean.tolist(), normalization_std=std.tolist(),
        standardized_fitted_features=Z.tolist(), duration_grid=list(DURATIONS),
        fitted_source_uids=sorted({r[k] for r in fitted for k in ('left_source_uid', 'right_source_uid') if r[k] is not None}),
        normalization_population='FINITE_LABELLED_DEV_FIT_ROWS_ONLY',
        fitting_calls=1, neural_training_steps=0, optimizer_calls=0,
        evaluation_labels_read=False, fallback_allowed=False, can_route=False)
    model['policy_sha256'] = digest(model)
    return model


def digest(model):
    payload = {k: v for k, v in model.items() if k != 'policy_sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf8')).hexdigest()


def predict(model, features):
    """No candidate outcomes are accepted by this API; failure has no fallback."""
    if model.get('policy_sha256') != digest(model):
        raise ValueError('Serialized duration policy hash mismatch')
    if model.get('kind') != 'DEV_ONLY_STANDARDIZED_1NN_V1':
        raise ValueError('Unexpected duration policy kind')
    x = _features(features)
    result = dict(gap_frames=None, status='NOT_PREDICTED', nearest_development_key=None,
                  standardized_squared_distance=None, fallback_used=False,
                  candidate_outcomes_read=False, policy_sha256=model['policy_sha256'])
    if model['status'] != 'FIT':
        result['status'] = 'MODEL_NOT_FIT'
        return result
    if x is None:
        result['status'] = 'BASELINE48_FEATURES_UNAVAILABLE'
        return result
    z = (x-np.asarray(model['normalization_mean']))/np.asarray(model['normalization_std'])
    distances = np.square(np.asarray(model['standardized_fitted_features'])-z).sum(axis=1)
    i = min(range(len(distances)), key=lambda j: (float(distances[j]), model['fitted_rows'][j]['key']))
    neighbor = model['fitted_rows'][i]
    result.update(gap_frames=int(neighbor['shortest_gap']), status='PREDICTED_UNSCREENED',
                  nearest_development_key=neighbor['key'], standardized_squared_distance=float(distances[i]))
    return result


def select(name, records=None, model=None, features=None):
    """Choose requested length; qualification remains the selected G1 result."""
    if name not in NAMES:
        raise ValueError('Unregistered duration policy')
    if name in ('Fixed48', 'Fixed24'):
        return dict(policy=name, gap_frames=48 if name == 'Fixed48' else 24,
                    status='REQUESTED_UNSCREENED', fallback_used=False)
    if name == 'ShortestScreened':
        gap = shortest_screened(records or [])
        return dict(policy=name, gap_frames=gap,
                    status='FIRST_PASSING_G1' if gap is not None else 'NO_PASSING_G1', fallback_used=False)
    if records is not None:
        raise ValueError('BoundaryConditioned must not read EVAL candidate outcomes')
    if model is None:
        raise ValueError('Sealed development duration model required')
    return dict(policy=name, **predict(model, features))
