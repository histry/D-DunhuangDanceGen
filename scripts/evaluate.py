import argparse,json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dunhuang.io import write,sha
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--compare');ap.add_argument('--video');ap.add_argument('--audio')
args=ap.parse_args();folder=Path(args.run).resolve()
request=json.loads((folder/'request.json').read_text());physical=json.loads((folder/'physical.json').read_text())
plan=json.loads((folder/'event_plan.json').read_text()) if (folder/'event_plan.json').exists() else None
summary=dict(frames=request['frames'],motion_sha256=sha(folder/'native.npy'),physical_ok=physical['absolute']['ok'],
    physical_reasons=physical['absolute']['reasons'],fk_error_m=physical['fk_max_abs_m'],
    events=len(plan['events']) if plan else 0,planned_time_coverage=1. if plan else 0.,
    controlled_time_fraction=plan['controlled_frames']/plan['frames'] if plan else 0.,
    unique_sources=len({e['source_uid'] for e in plan['events']}) if plan else 0,
    unique_events=len({e['event_uid'] for e in plan['events']}) if plan else 0,
    event_semantic_quality='NOT_INDEPENDENTLY_EVALUATED',naturalness='NOT_HUMAN_EVALUATED')
if request.get('mode')=='event_guided':
    audit=json.loads((folder/'soft_reference_audit.json').read_text())
    summary.update(controlled_time_fraction=None,conditioning_kind='continuous_soft_tokens_not_hard_output_constraint',
        soft_reference_time_fraction=audit['supported_time_frames']/request['frames'],
        soft_reference_joint_fraction=audit['supported_fraction'])
if args.compare:summary['differs_from_comparison']=sha(folder/'native.npy')!=sha(Path(args.compare)/'native.npy')
write(folder/'summary.json',summary)
if args.video and args.compare:
    cmd=[sys.executable,str(ROOT/'scripts/preview_pair.py'),args.compare,str(folder),args.video,'--right-label','Event Adapter (research)' if request.get('mode')=='event_guided' else 'Event Plan + Temporal']
    if args.audio:cmd+=['--audio',args.audio]
    subprocess.run(cmd,check=True)
print(json.dumps(summary,indent=2))
