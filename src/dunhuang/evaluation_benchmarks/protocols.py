"""Identity-bound ground/skeleton protocols and immutable Legacy provenance."""
from dataclasses import dataclass,asdict
import hashlib,json
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class GroundAndSkeletonContract:
    skeleton_name: str
    skeleton_sha256: str
    floor_y_m: float
    floor_source: str
    floor_reference_sha256: str
    frames: int
    fps: int = 30
    units: str = 'm'
    up_axis: str = 'y'
    ground_truth: bool = False

    def __post_init__(self):
        valid_hash=lambda value:isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
        if (not self.skeleton_name or not valid_hash(self.skeleton_sha256) or
            not np.isfinite(self.floor_y_m) or self.frames<4 or self.fps!=30 or
            self.units!='m' or self.up_axis!='y' or not valid_hash(self.floor_reference_sha256)):
            raise ValueError('Ground/skeleton identity, SI and clock required')
        allowed={'measured_scene','frozen_common_reference','estimated_candidate_floor'}
        if self.floor_source not in allowed:
            raise ValueError('Unknown floor protocol')
        if self.ground_truth and self.floor_source!='measured_scene':
            raise ValueError('Estimated/common ground is not measured ground truth')

    def record(self):
        value=dict(schema='ground_and_skeleton_contract_v1',**asdict(self))
        return dict(value,id=digest(value),
                    confidence='EXTERNAL_CALIBRATION_REQUIRED' if self.ground_truth else 'UNVALIDATED_GEOMETRIC_PROXY')


def integrity(motion,joints,contract,official_fk=None,tolerance=1e-5):
    """A mathematical representation gate, separate from empirical limits."""
    x=np.asarray(motion);j=np.asarray(joints)
    reasons=[]
    if x.shape!=(contract.frames,151) or x.dtype!=np.float32:reasons.append('INVALID_MOTION_LAYOUT')
    if j.shape!=(contract.frames,24,3):reasons.append('INVALID_FK_LAYOUT')
    if reasons:return dict(status='INVALID',ok=False,reasons=reasons)
    if not np.isfinite(x).all() or not np.isfinite(j).all():
        return dict(status='INVALID',ok=False,reasons=['NONFINITE_VALUES'])
    r=x[:,7:].reshape(-1,24,6).astype(float);a=r[...,:3];b=r[...,3:]
    an=np.linalg.norm(a,axis=-1,keepdims=True);unit=a/np.maximum(an,1e-12)
    orth=b-(unit*b).sum(-1,keepdims=True)*unit
    if np.any(an<1e-6) or np.any(np.linalg.norm(orth,axis=-1)<1e-6):
        reasons.append('DEGENERATE_ROT6D')
    error=None
    if official_fk is not None:
        o=np.asarray(official_fk)
        if o.shape!=j.shape or not np.isfinite(o).all():reasons.append('INVALID_OFFICIAL_FK')
        else:
            error=float(abs(j-o).max())
            if error>tolerance:reasons.append('FK_IDENTITY_MISMATCH')
    return dict(status='VALID' if not reasons else 'INVALID',ok=not reasons,reasons=reasons,
                official_fk_max_abs_m=error,fk_tolerance_m=tolerance,
                conditioning_proxy_is_not_singularity_test=True)


def threshold_registry(limits):
    rows=[]
    for key,value in limits.items():
        if key in ('rot6d_nonfinite_ratio','rot6d_degenerate_ratio'):
            category='A_representation'
        elif key.startswith('root_horizontal_') or key=='root_y_robust_range_m':
            category='C_scene_task'
        else:category='B_plausibility_proxy'
        units=('m/s^3' if 'jerk' in key else 'rad/s^2' if 'angular_acceleration' in key else
               'rad' if 'rotation_step' in key else 'm/s' if 'mps' in key or 'speed' in key else
               'dimensionless' if any(s in key for s in ('ratio','collinearity')) else 'm')
        rows.append(dict(metric=key,units=units,category=category,legacy_threshold=float(value),
            direction='lower' if key=='foot_penetration_min_m' else 'upper',
            metric_definition=dict(legacy_key=key,operator='pinned Native source implementation',
                aggregation='max of1s-windowP95' if 'window' in key else 'P95' if 'p95' in key else
                            'maximum' if 'max' in key else 'legacy named statistic',
                reference='motion_geometry/physical.py at pinned legacy commit',
                support_population='derived-contact OR filtered low-height, provisionally static; not independent static truth' if key.startswith('foot_') else None),
            threshold_origin='project_legacy_engineering_default',
            paper_universal_threshold=None,calibration_status='NOT_CALIBRATED',
            contact_source='rederived_FK_union_low_height' if key.startswith('foot_') else None,
            applicability='explicit scene task' if category.startswith('C') else 'Legacy protocol only; empirical validity unresolved',
            legacy_gate_unchanged=True))
    return dict(schema='native_threshold_provenance_v1',contract='Native-Legacy-v1',
                count=len(rows),metrics=rows,legacy_gate_unchanged=True)


def shadow_decision(numerical,quality_evidence=False,contact_evidence=False):
    return dict(contract='Native-Calib-v2',mode='SHADOW',operational_qualified=False,
        result='INVALID_REPRESENTATION' if not numerical['ok'] else 'UNKNOWN_UNQUALIFIED',
        legacy_result_overridden=False,can_route=False,
        blockers=[] if not numerical['ok'] else
          ['UNVALIDATED_CONTACT_AND_QUALITY_LABELS','UNQUALIFIED_THRESHOLD_CALIBRATION'],
        quality_evidence_available=bool(quality_evidence),contact_evidence_available=bool(contact_evidence),
        human_naturalness='NOT_EVALUATED',dynamic_feasibility='NOT_SIMULATED')
