"""Publish aggregate measurements only; raw identities and machine paths stay private."""
import argparse,json,sys,subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import sha,write


def read(p):return json.loads(Path(p).read_text())


def stats(values):
    valid=[float(v) for v in values if v is not None]
    return dict(samples=len(valid),not_evaluable=len(values)-len(valid),
        median=float(np.median(valid)) if valid else None,
        min=min(valid) if valid else None,max=max(valid) if valid else None)


def summarize(root,replay_path):
    root=Path(root);status=read(root/'status.json');replay=read(replay_path)
    if status['status']!='COMPLETED_NATIVE_RELIABILITY_SHADOW_AUDIT' or replay['status']!='FRESH_NATIVE_RELIABILITY_REPLAY_VERIFIED':
        raise ValueError('Completed experiment and independent replay required')
    if replay['source_protocol_sha256']!=sha(root/'protocol.json'):
        raise ValueError('Replay belongs to another source protocol')
    protocol=read(root/'protocol.json');history=read(root/'historical_inventory.json')
    training=read(root/'train/summary.json');calibration=read(root/'calibration.json')
    evidence=read(root/'labelled_candidates.json')
    generated=read(root/'generated/results.json')
    cohorts=defaultdict(list)
    for row in generated:cohorts[(row['domain'],row['mode'])].append(row)
    cohort_summary=[]
    for (domain,mode),rows in sorted(cohorts.items()):
        continuous=[r['analysis']['layers']['continuous'] for r in rows]
        cohort_summary.append(dict(domain=domain,mode=mode,executions=len(rows),
            unique_motions=len({r['motion']['sha256'] for r in rows}),
            legacy_passes=sum(r['analysis']['legacy']['ok'] for r in rows),
            pfc=stats([v['pfc']['value'] for v in continuous]),
            upperbody_radians_per_second=stats([v['activity']['upperbody']['mean_radians_per_second'] for v in continuous]),
            joint_speed_mps=stats([v['activity']['joint_speed']['mean'] for v in continuous]),
            raw_jerk_max_mps3=stats([v['jerk']['raw']['max'] for v in continuous]),
            independent_static_speed=stats([v['contact']['independent_static_speed']['p95'] for v in continuous])))
    sensitivity=read(root/'sensitivity/generated_ground_skeleton.json')
    groups=defaultdict(list)
    for row in sensitivity:groups[(row['domain'],row['mode'],row['protocol'])].append(row)
    geometry=[]
    for (domain,mode,plane),rows in sorted(groups.items()):
        geometry.append(dict(domain=domain,mode=mode,ground_protocol=plane,executions=len(rows),
            penetrate_mm=stats([r['neutral_mesh']['penetrate_mm'] for r in rows]),
            float_mm=stats([r['neutral_mesh']['float_mm'] for r in rows]),
            geometric_skate_mm=stats([r['neutral_mesh']['skate_mm'] for r in rows]),
            foot_geometry_max_abs_m=stats([r['foot_geometry_max_abs_m'] for r in rows])))
    sources=read(root/'sensitivity/source_duration_ground_noise.json')
    source_groups=defaultdict(list);skipped=0
    for row in sources:
        if row['status']=='INSUFFICIENT_REAL_FRAMES_NO_PADDING':skipped+=1
        elif row['status']=='COMPUTED':source_groups[('duration_frames',row['length_frames'])].append(row)
        elif row['status']=='NOISE_SENSITIVITY':source_groups[('noise_amplitude_m',row['jitter_amplitude_m'])].append(row)
    source_summary=[]
    for (kind,value),rows in sorted(source_groups.items()):
        source_summary.append(dict(variable=kind,value=value,recordings=len(rows),
            jerk_max_mps3=stats([r['jerk']['raw']['max'] for r in rows]),
            jerk_p95_mps3=stats([r['jerk']['raw']['p95'] for r in rows]),
            jerk_violation_fraction=stats([r['jerk']['violation_fraction'] for r in rows]),
            legacy_passes=sum(r['legacy']['ok'] for r in rows)))
    long_rows=read(root/'sensitivity/saved_long_prefixes.json')
    long_summary=[dict(cohort=Path(r['run']).name,prefix_frames=r['prefix_frames'],
        raw_jerk=r['jerk']['raw'],violation_fraction=r['jerk']['violation_fraction'],
        longest_stencil_seconds=r['jerk']['longest_consecutive_stencil_seconds'],legacy_pass=r['legacy']['ok']) for r in long_rows]
    controls=read(root/'controlled/report.json')
    control_summary=[dict(case=r['case'],amplitude=r['amplitude'],legacy_pass=r['legacy']['ok'],
        legacy_floor_y_m=r['legacy_floor_y_m'],legacy_foot_penetration_min_m=r['legacy_foot_penetration_min_m'],
        mesh_penetrate_mm=r['mesh']['penetrate_mm'],mesh_float_mm=r['mesh']['float_mm'],
        synthetic_static_p95_mps=r['independent_synthetic_contact']['independent_static_speed']['p95']) for r in controls['cases']]
    parity=read(root/'sensitivity/mesh_reference_parity.json')
    provenance=read(root/'threshold_provenance.json')
    return dict(schema='public_native_reliability_aggregate_v1',experiment_code_commit=protocol['code_commit'],
        summary_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        mode='SHADOW_UNQUALIFIED',operational_qualified=False,can_route=False,
        legacy_identity=read(root/'legacy_identity.json'),threshold_provenance=provenance,
        historical=dict(records=history['record_count'],unique_measurements=history['unique_count'],
            skipped_different_or_unsealed=len(history['skipped']),legacy_replay_exact=True),
        train=training,calibration=calibration,labelled_candidates=evidence,
        generated=cohort_summary,ground_skeleton=geometry,source_duration_noise=source_summary,
        skipped_source_lengths=skipped,long_prefixes=long_summary,
        controlled=dict(count=len(controls['cases']),curves=controls['curves'],cases=control_summary),
        mesh_reference_parity={k:v for k,v in parity.items() if k!='model'},fresh_replay=replay,
        input_receipt_hashes={name:sha(root/name) for name in (
            'protocol.json','historical_inventory.json','train/windows.jsonl','train/calibration_rows.json',
            'calibration.json','labelled_candidates.json','generated/results.json',
            'sensitivity/generated_ground_skeleton.json','sensitivity/source_duration_ground_noise.json',
            'sensitivity/saved_long_prefixes.json','controlled/report.json')},
        replay_receipt_sha256=sha(replay_path),annotation_packet_records=len(read(root/'annotation_review/index.json')['records']),
        expert_labels_supplied=False,training_steps=0,optimizer_calls=0,new_generations=0,
        boundaries=['Fitted source motion is not independently certified high-quality truth.',
            'Reference exceedance is not false rejection. Conditional real static metrics are NOT_EVALUABLE without labels.',
            'Fixed reference ground is a geometric proxy, not measured scene ground.',
            'Six connected source groups do not establish independent statistical coverage.',
            'Controlled neutral fixtures test operators, not real dance quality or detection accuracy.',
            'PhysDiff is a declared Appendix reimplementation, without author-code or leaderboard parity.',
            'Legacy kinematic/contact audit does not simulate forces, friction or balance.'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--replay',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    result=summarize(a.root,a.replay)
    serialized=json.dumps(result,ensure_ascii=False)
    if any(text in serialized for text in ('/home/','/storage/','recording_uid','window_uid','choreography_uid')):
        raise ValueError('Raw source identifiers or machine paths must not be published')
    write(out,result)
    print('AGGREGATE_WRITTEN',out)

if __name__=='__main__':main()
