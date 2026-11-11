"""Fresh read-only replay of Native reliability artifacts, never a new threshold fit."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import checked,write,sha
from dunhuang.evaluation_benchmarks.native_shadow import LegacyBackend
from dunhuang.evaluation_benchmarks.calibration import reference_report,jerk_report
from dunhuang.evaluation_benchmarks.contact_evidence import load_evidence,conditioned_metrics
from dunhuang.evaluation_benchmarks.smpl_mesh import SMPLMesh
from dunhuang.evaluation_benchmarks.physdiff_metrics import population_metrics
from dunhuang.evaluation_benchmarks.benchmark_report import official_mesh_parity
from dunhuang.evaluation_benchmarks.protocols import digest


def read(p):return json.loads(Path(p).read_text())


def main():
    import torch
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();root=Path(a.root);out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    protocol=read(root/'protocol.json');assets=read(checked(protocol['assets']))
    config=read(checked(protocol['config']));mesh_assets=read(checked(protocol['mesh_assets']))
    backend=LegacyBackend(assets)
    assert backend.identity==read(root/'legacy_identity.json')
    assert backend.registry==read(root/'threshold_provenance.json')
    inventory=read(root/'historical_inventory.json');seen=set()
    for r in inventory['records']:
        for value in r['files'].values():checked(value)
        if r['identity'] in seen:continue
        seen.add(r['identity']);x=np.load(checked(r['files']['native.npy']));fk=np.load(checked(r['files']['native_fk.npy']))
        original=read(checked(r['files']['physical.json']))
        audit=backend.audit(x,backend.cfg,precomputed_joints=fk)
        assert audit==original['audit'] and backend.gate(audit)==original['absolute']
    ground=read(root/'train/frozen_ground_anchors.json');count=0;fresh_records=[]
    for row in map(json.loads,(root/'train/windows.jsonl').read_text().splitlines()):
        x=np.load(checked(row['motion']));anchor=ground[row['recording_uid']]
        labels,_=load_evidence(row.get('labels'),row['motion']['sha256'],len(x))
        result,_=backend.analyse(x,row['motion']['sha256'],anchor['floor_y_m'],
            anchor['reference']['sha256'],labels=labels,expected=row['legacy'])
        assert result['legacy_contact']==row['legacy_contact']
        assert result['layers']==row['layers']
        assert result['layers']['judgments']['calibrated']['can_route'] is False
        lineage={k:row[k] for k in ('window_uid','role','recording_uid','choreography_uid','source_group','music_uid','start_frame')}
        metrics={spec['metric']:result['legacy_audit'].get(spec['metric']) for spec in backend.registry['metrics']}
        if not result['legacy_contact']['sample_count']:
            for key in ('foot_skate_mps_p95','foot_skate_mps_max','foot_support_drift_m_p95','foot_support_drift_m_max'):metrics[key]=None
        contact=result['layers']['continuous']['contact'];candidate=dict(metrics)
        for key,field,stat in [('foot_skate_mps_p95','independent_static_speed','p95'),
                ('foot_skate_mps_max','independent_static_speed','max'),
                ('foot_support_drift_m_p95','valid_static_drift','p95'),
                ('foot_support_drift_m_max','valid_static_drift','max')]:
            candidate[key]=contact[field][stat]
        for key in ('foot_contact_height_m_max','foot_penetration_min_m'):candidate[key]=None
        fresh_records.append(dict(lineage,metrics=metrics,candidate_metrics=candidate))
        count+=1
    # Source metrics and grouped resampling use the bound same deterministic protocol.
    records=read(root/'train/calibration_rows.json');assert records==fresh_records
    fresh=reference_report(records,backend.registry,config['bootstrap']['iterations'],config['bootstrap']['seed'])
    assert fresh==read(root/'calibration.json')
    gen=read(root/'generated/results.json')
    for r in gen:
        x=np.load(checked(r['motion']));fk=np.load(checked(r['fk']))
        result,_=backend.analyse(x,r['motion']['sha256'],r['anchor']['value'],
            r['anchor']['motion']['sha256'],fk=fk,expected=r['analysis']['legacy'])
        assert result==r['analysis']
        case=root/'generated'/r['name']
        original=read(case/'physical.json')
        assert original['absolute']==result['legacy']
        sidecar=read(case/'physical_protocol.json')
        assert sidecar['motion_sha256']==r['motion']['sha256']
        assert sidecar['audit_sha256']==digest(original['audit'])
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    mesh=SMPLMesh(mesh_assets['mesh_model'],mesh_assets.get('mesh_device','cpu'))
    sensitivity=read(root/'sensitivity/generated_ground_skeleton.json')
    cached={};mesh_count=0
    for r in sensitivity:
        key=r['motion']['sha256']
        if key not in cached:
            x=np.load(checked(r['motion']));minimum,joints=mesh.sequence(x)
            saved=np.load(checked(r['mesh_cache']))
            assert np.array_equal(minimum,saved['minimum_y']) and np.array_equal(joints,saved['joints'])
            cached[key]=(minimum,joints);mesh_count+=1
        minimum,joints=cached[key]
        assert population_metrics(minimum,joints,r['floor_y_m'],contact_height=.05)==r['neutral_mesh']
    # Independently reconstruct the raw-source prefixes, fixed planes and noise copies.
    from dunhuang.evaluation_benchmarks.native_sensitivity import sensitivity as run_sensitivity
    source_protocol=read(checked(protocol['prior_training_protocol']))
    manifest=read(checked(assets['paired_generation_manifest']))
    by_uid={r['window_uid']:r for r in manifest['rows']}
    packet=read(root/'annotation_review/index.json')
    selected=[by_uid[r['window_uid']] for r in packet['records']]
    replay_dir=out.with_suffix('.sensitivity')
    run_sensitivity(backend,selected,gen,mesh_assets,config,replay_dir)
    for name in ('source_duration_ground_noise.json','saved_long_prefixes.json','mesh_reference_parity.json'):
        assert read(root/'sensitivity'/name)==read(replay_dir/name)
    controlled=read(root/'controlled/report.json')
    for r in controlled['cases']:
        folder=root/'controlled'/r['case'];x=np.load(folder/'native.npy')
        assert sha(folder/'native.npy')==r['motion_sha256']
        fk=backend.fk(x);minimum,joints=mesh.sequence(x)
        assert backend.gate(backend.audit(x,backend.cfg,precomputed_joints=fk))==r['legacy']
        assert population_metrics(minimum,joints,r['fixed_floor_y_m'],contact_height=.05)==r['mesh']
        labels=np.load(folder/'labels.npy')
        assert conditioned_metrics(fk,r['fixed_floor_y_m'],labels=labels)==r['independent_synthetic_contact']
        assert jerk_report(fk,backend.limits['joint_jerk_mps3_max'])==r['jerk']
    assert all(c['nondecreasing'] for c in controlled['curves'].values())
    status=read(root/'status.json');assert status['legacy_results_unchanged'] and not status['operational_qualified']
    write(out,dict(status='FRESH_NATIVE_RELIABILITY_REPLAY_VERIFIED',train_windows=count,
        historical_unique_replays=len(seen),generated_replays=len(gen),mesh_geometry_replays=mesh_count,
        controlled_replays=len(controlled['cases']),source_sensitivity_replayed=True,\n        calibration_rows_reconstructed=True,legacy_results_unchanged=True,
        source_protocol_sha256=sha(root/'protocol.json'),optimizer_calls=0,new_generations=0,
        operational_qualified=False,mode='SHADOW',expert_labels_supplied=False))
    print('NATIVE_RELIABILITY_VERIFIED',count,len(gen),mesh_count,flush=True)

if __name__=='__main__':main()
