"""Train-only descriptive Native audit. Never filters data or selects weights."""
import argparse,json,sys,time,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import checked,pinned,write,sha
from dunhuang.research.event_metrics import dynamics


def read(path):return json.loads(Path(path).read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--training',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();assets=read(a.assets);training=Path(a.training)
    protocol=read(training/'protocol.json');status=read(training/'status.json')
    if sha(training/'protocol.json')!=status['protocol_sha256'] or protocol['paired_manifest']!=assets['paired_generation_manifest']:
        raise ValueError('Training input lineage differs')
    tasks=read(training/'source_pairs.json')
    manifest=read(checked(assets['paired_generation_manifest']))
    from dunhuang.generation.source_pairs import source_pairs
    if tasks!=source_pairs(manifest):
        raise ValueError('Cross-recording task population changed')
    by_uid={r['window_uid']:r for r in manifest['rows']}
    selected=sorted({p['target_uid'] for p in tasks if p['role']=='train'})
    if len(selected)!=protocol['train_tasks'] or any(by_uid[u]['role']!='train' for u in selected):
        raise ValueError('Train-only audit population differs')
    root=pinned(assets['legacy_root'],assets['legacy_commit']);sys.path.insert(0,str(root))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    from motion.geometry import fk_24_np
    from motion.physics import audit_motion_np
    from contracts.physical_quality import evaluate_physical_audit
    write(out/'protocol.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        training_protocol_sha256=sha(training/'protocol.json'),source_tasks_sha256=sha(training/'source_pairs.json'),paired_manifest=assets['paired_generation_manifest'],
        legacy_commit=assets['legacy_commit'],population='all cross-recording Train targets; no Val/Test',
        targets=len(selected),optimizer_calls=0,data_filtering=False,checkpoint_selection=False,
        scope='current Native contract on fitted targets, not source invalidity or human plausibility labels'))
    reasons=Counter();passed=0;peaks=[];activities=[]
    with (out/'target_audits.jsonl').open('w',encoding='utf8') as stream:
        for i,uid in enumerate(selected,1):
            row=by_uid[uid];native=np.load(checked(row['motion']),allow_pickle=False)
            if native.dtype!=np.float32 or native.shape!=(150,151):raise ValueError('Full real target required')
            fk=fk_24_np(native);audit=audit_motion_np(native,precomputed_joints=fk)
            gate=evaluate_physical_audit(audit);metric=dynamics(native,fk)
            stream.write(json.dumps(dict(window_uid=uid,motion=row['motion'],role='train',physical=gate,dynamics=metric),allow_nan=False)+'\n')
            passed+=bool(gate['ok']);reasons.update(gate['reasons'])
            peaks.append(metric['jerk_peak_mps3']);activities.append(metric['upper8_angular_speed_mean_radps'])
            if i%100==0 or i==len(selected):
                stream.flush();write(out/'progress.json',dict(completed=i,total=len(selected),native_passes=passed,seconds=time.monotonic()-started))
                print('TRAIN_TARGET_AUDIT',i,len(selected),passed,flush=True)
    write(out/'summary.json',dict(status='COMPLETED_TRAIN_ONLY_NATIVE_DESCRIPTION',
        targets=len(selected),native_passes=passed,native_failures=len(selected)-passed,reason_counts=dict(reasons),
        jerk_peak_mps3=dict(min=min(peaks),median=float(np.median(peaks)),max=max(peaks)),
        upper8_angular_speed_mean_radps=dict(min=min(activities),median=float(np.median(activities)),max=max(activities)),
        target_audits_sha256=sha(out/'target_audits.jsonl'),optimizer_calls=0,new_weight_updates=0,
        data_filtering=False,checkpoint_selection=False,seconds=time.monotonic()-started,
        interpretation='Native contract membership only; failures do not label recordings invalid or exclude their source events'))
    print('TRAIN_TARGET_AUDIT_COMPLETED',passed,len(selected),flush=True)

if __name__=='__main__':main()
