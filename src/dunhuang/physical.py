"""Subprocess entry: pinned legacy Native is the only physical authority."""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .io import pinned, write
from .contracts import MotionAdapter, Q
from .evaluation import evaluate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('request');a=ap.parse_args()
    q=json.loads(Path(a.request).read_text());root=pinned(q['legacy_root'],q['legacy_commit'])
    sys.path.insert(0,str(root))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    from motion.geometry import fk_24_np
    from motion.physics import audit_motion_np
    from contracts.physical_quality import evaluate_physical_audit, PhysicalQualityLimits
    out=Path(q['output']);x=MotionAdapter.to_native(np.load(out/'edge.npy')[:q['frames']])
    positions=fk_24_np(x);official=np.load(out/'official_fk.npy')[:len(x)]@Q
    error=float(abs(positions-official).max())
    if error>1e-5:
        raise ValueError('Official/native FK mismatch: '+str(error))
    audit=audit_motion_np(x,precomputed_joints=positions);gate=evaluate_physical_audit(audit)
    np.save(out/'native.npy',x);np.save(out/'native_fk.npy',positions)
    write(out/'physical.json',dict(audit=audit,absolute=gate,limits=asdict(PhysicalQualityLimits.from_environment()),
        fk_max_abs_m=error,legacy_commit=q['legacy_commit'],relative_parent_caps=False,
        full_generation_qualified=False))
    from .evaluation_benchmarks.native_shadow import write_legacy_sidecar
    write_legacy_sidecar(q,out,audit,gate,error)
    write(out/'evaluation.json',evaluate(x,positions,q.get('anchors',[]),q.get('beat_frames',[])))
    print('NATIVE',gate['ok'],'FK_ERROR',error,flush=True)
if __name__=='__main__':main()
