"""CPU-only Event Plan -> source-backed soft prompt preflight.

Does not run EDGE, train an Adapter, or certify any generated motion.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from dunhuang.io import sha, write
from dunhuang.planning.event_plan import validate
from dunhuang.generation.soft_reference import build_soft_reference


def inspect(plan_path, scaler_path, output):
    plan_path = Path(plan_path).resolve()
    scaler_path = Path(scaler_path).resolve()
    output = Path(output).resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    validate(plan, plan['audio_sha256'], plan['frames'])
    with np.load(scaler_path, allow_pickle=False) as scaler:
        values, mask, phase, audit = build_soft_reference(
            plan, plan['frames'], scaler['scale'], scaler['bias'])
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'soft_reference.npz',
                        value=values, known=mask, source_phase=phase)
    write(output/'status.json', dict(
        status='SOFT_REFERENCE_PREFLIGHT_ONLY' if mask.any() else 'NO_SUPPORTED_SOFT_REFERENCE',
        training_steps=0, native_executed=False, native_ok=None,
        adapter_trained=False, music_qualified=False, physical_qualified=False,
        plan_sha256=sha(plan_path), scaler_sha256=sha(scaler_path),
        soft_reference_sha256=sha(output/'soft_reference.npz'), audit=audit))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--scaler', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    print(inspect(args.plan, args.scaler, args.out))
