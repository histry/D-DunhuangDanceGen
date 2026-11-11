"""Fresh read-only replay of Native reliability artifacts, never a new threshold fit."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import checked,write,sha
from dunhuang.evaluation_benchmarks.native_shadow import LegacyBackend
from dunhuang.evaluation_benchmarks.calibration import reference_report
from dunhuang.evaluation_benchmarks.contact_evidence import load_evidence
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
    ground=read(root/'train/frozen_ground_anchors.json');count=0
    for row in map(json.loads,(root/'train/windows.jsonl').read_text().splitlines()):
        x=np.load(checked(row['motion']));anchor=ground[row['recording_uid']]
        labels,_=load_evidence(row.get('labels'),row['motion']['sha256'],len(x))
        result,_=backend.analyse(x,row['motion']['sha256'],anchor['floor_y_m'],
            anchor['reference']['sha256'],labels=labels,expected=row['legacy'])
        assert result['legacy_contact']==row['legacy_contact']
        assert result['layers']==row['layers']
        assert result['layers']['judgments']['calibrated']['can_route'] is False
        count+=1
    # Source metrics and grouped resampling use the bound same deterministic protocol.
    records=read(root/'train/calibration_rows.json')
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
    controlled=read(root/'controlled/report.json')
    for r in controlled['cases']:
        folder=root/'controlled'/r['case'];x=np.load(folder/'native.npy')
        assert sha(folder/'native.npy')==r['motion_sha256']
        fk=backend.fk(x);minimum,joints=mesh.sequence(x)
        assert backend.gate(backend.audit(x,backend.cfg,precomputed_joints=fk))==r['legacy']
        assert population_metrics(minimum,joints,r['fixed_floor_y_m'],contact_height=.05)==r['mesh']
    assert all(c['nondecreasing'] for c in controlled['curves'].values())
    status=read(root/'status.json');assert status['legacy_results_unchanged'] and not status['operational_qualified']
    write(out,dict(status='FRESH_NATIVE_RELIABILITY_REPLAY_VERIFIED',train_windows=count,
        historical_unique_replays=len(seen),generated_replays=len(gen),mesh_geometry_replays=mesh_count,
        controlled_replays=len(controlled['cases']),legacy_results_unchanged=True,
        source_protocol_sha256=sha(root/'protocol.json'),optimizer_calls=0,new_generations=0,
        operational_qualified=False,mode='SHADOW',expert_labels_supplied=False))
    print('NATIVE_RELIABILITY_VERIFIED',count,len(gen),mesh_count,flush=True)

if __name__=='__main__':main()
