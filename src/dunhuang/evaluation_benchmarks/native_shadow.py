"""Frozen Native arithmetic plus separately versioned read-only research layers."""
import sys,json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from dunhuang.io import pinned,sha,write
from .protocols import GroundAndSkeletonContract,integrity,threshold_registry,digest,shadow_decision
from .contact_evidence import conditioned_metrics,distribution,FEET
from .calibration import jerk_report
from .edge_pfc import pfc
from dunhuang.generation.diagnostics import upperbody_activity

SOURCE_FILES=('contracts/physical_quality.py','motion/physics.py','motion_geometry/physical.py',
              'motion_geometry/smpl24.py','configs/baseline.env')


class LegacyBackend:
    def __init__(self,assets):
        root=pinned(assets['legacy_root'],assets['legacy_commit']);sys.path.insert(0,str(root))
        from data_pipeline.prepare_domains import _baseline_environment
        _baseline_environment()
        from motion.geometry import fk_24_np
        from motion.physics import audit_motion_np,derive_contacts_np
        from motion.config import MotionGenerationConfig
        from contracts.physical_quality import evaluate_physical_audit,PhysicalQualityLimits
        from motion_geometry.smpl24 import skeleton_fingerprint
        from motion_geometry.physical import median_filter_bool_np,_odd_window,_foot_speed_mps
        self.fk=fk_24_np;self.audit=audit_motion_np;self.derive=derive_contacts_np
        self.gate=evaluate_physical_audit;self.cfg=MotionGenerationConfig()
        self.limits=asdict(PhysicalQualityLimits.from_environment())
        self.skeleton_hash=skeleton_fingerprint()
        self.low_filter=lambda x:median_filter_bool_np(x,_odd_window(1/12,self.cfg.fps))
        self.speed=lambda f:_foot_speed_mps(f,self.cfg.fps)
        self.identity=dict(contract='Native-Legacy-v1',legacy_commit=assets['legacy_commit'],
            code_sha256={p:sha(root/p) for p in SOURCE_FILES},
            skeleton_sha256=self.skeleton_hash,limits=self.limits,motion_config=asdict(self.cfg),
            fps=30,units='SI',support_policy='final_fail_closed',
            metric_schema='dunhuang_physical_metrics_si_v5_contact_states',
            scope='kinematic and contact plausibility; not dynamics simulation')
        self.identity=json.loads(json.dumps(self.identity,allow_nan=False))
        self.identity['id']=digest(self.identity)
        self.registry=threshold_registry(self.limits)

    def analyse(self,motion,motion_sha,floor=None,floor_reference=None,labels=None,fk=None,expected=None):
        x=np.asarray(motion)
        fk=self.fk(x) if fk is None else np.asarray(fk,dtype=np.float32)
        audit=self.audit(x,self.cfg,precomputed_joints=fk);gate=self.gate(audit)
        if expected is not None and gate!=expected:
            raise ValueError('Immutable Legacy Native differs from stored target receipt')
        derived,confidence,legacy_floor,feet=self.derive(x,self.cfg,precomputed_joints=fk)
        low=self.low_filter(feet[:,:,1]<=legacy_floor+.055)
        union=derived|low;speed=self.speed(feet)
        # Keep the exact Legacy sample population, including its initial zeros.
        values=speed[union].astype(float)
        p95=float(np.percentile(values,95)) if len(values) else 0.
        maximum=float(values.max()) if len(values) else 0.
        if p95!=audit['foot_skate_mps_p95'] or maximum!=audit['foot_skate_mps_max']:
            raise ValueError('Legacy contact responsibility population differs')
        exceeding=(speed>self.limits['foot_skate_mps_p95'])&union
        responsibility=dict(sample_count=int(union.sum()),possible_samples=int(union.size),
            legacy_p95_mps=p95,legacy_max_mps=maximum,reconstruction_exact=True,
            exceeding_frame_foot_observations=int(exceeding.sum()),
            exceedance_ownership={name:int((exceeding&m).sum()) for name,m in {
                'derived_only':derived&~low,'height_only':low&~derived,'both':derived&low}.items()},
            semantic_static_truth='UNKNOWN_WITHOUT_INDEPENDENT_LABELS',
            original_prediction_ignored_by_legacy_gate=True,
            populations=conditioned_metrics(fk,legacy_floor,derived=derived,low=low,predicted=x[:,:4]>=.5))
        common=legacy_floor if floor is None else float(floor)
        source='estimated_candidate_floor' if floor is None else 'frozen_common_reference'
        contract=GroundAndSkeletonContract('Legacy SMPL24 fixed offsets',self.skeleton_hash,common,source,
            motion_sha if floor_reference is None else floor_reference,len(x))
        numeric=integrity(x,fk,contract)
        if not numeric['ok']:raise ValueError('Invalid numerical representation')
        newlow=self.low_filter(feet[:,:,1]<=common+.055)
        contacts=conditioned_metrics(fk,common,labels=labels,low=newlow,predicted=x[:,:4]>=.5)
        layer=dict(schema='native_three_layer_audit_v1',motion_sha256=motion_sha,
            ground_skeleton=contract.record(),numerical_integrity=numeric,
            continuous=dict(activity=dict(upperbody=upperbody_activity(x),
                joint_speed=distribution(np.linalg.norm(np.diff(fk,axis=0)*30,axis=-1),(len(x)-1)*24,'m/s'),
                static_motion_is_not_automatically_high_quality=True),
                contact=contacts,jerk=jerk_report(fk,self.limits['joint_jerk_mps3_max']),pfc=pfc(fk),
                root_motion={k:v for k,v in audit.items() if k.startswith('root_')},
                foot_joint_height_relative_m=distribution((feet[:,:,1]-common).reshape(-1),feet.shape[0]*4,'m'),
                mesh_geometry=dict(status='NOT_COMPUTED_IN_JOINT_ONLY_LAYER',penetrate_mm=None,float_mm=None),
                full_physical_simulation='NOT_PERFORMED'),
            judgments=dict(legacy=dict(version='Native-Legacy-v1',result='PASS' if gate['ok'] else 'FAIL',
                contract_id=self.identity['id']),calibrated=shadow_decision(numeric,contact_evidence=labels is not None)),
            raw_vs_independent_semantics_separate=True)
        return dict(legacy=gate,legacy_audit=audit,legacy_contact=responsibility,layers=layer),fk


def write_legacy_sidecar(q,out,audit,gate,fk_error):
    """Append metadata only. physical.json and its verdict keep the old wire form."""
    out=Path(out);backend=LegacyBackend(q)
    if backend.limits!=gate['limits']:raise ValueError('Legacy provenance limit mismatch')
    write(out/'physical_protocol.json',dict(schema='legacy_native_identity_sidecar_v1',
        legacy=backend.identity,motion_sha256=sha(out/'native.npy'),fk_sha256=sha(out/'native_fk.npy'),
        audit_sha256=digest(audit),gate_sha256=digest(gate),official_fk_max_abs_m=float(fk_error),
        ground_source='estimated_candidate_floor',floor_ground_truth=False,
        calibration_contract='Native-Calib-v2',calibration_mode='SHADOW_UNQUALIFIED',can_route_shadow=False,
        legacy_physical_file_unchanged=True))

    if q['frames']>=4:
        x=np.load(out/'native.npy');fk=np.load(out/'native_fk.npy')
        layers,_=backend.analyse(x,sha(out/'native.npy'),fk=fk,expected=gate)
        write(out/'native_layers.json',layers['layers'])
    else:
        write(out/'native_layers.json',dict(schema='native_three_layer_audit_v1',
            motion_sha256=sha(out/'native.npy'),status='NOT_EVALUABLE_SHORT_SEQUENCE',
            judgments=dict(calibrated=dict(contract='Native-Calib-v2',mode='SHADOW',can_route=False,
                result='UNKNOWN_UNQUALIFIED')),legacy_result_unchanged=True))
