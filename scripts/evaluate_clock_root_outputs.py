"""Aggregate physical/source-agreement and real Shadow inference, without selection."""
import argparse,json,sys,subprocess
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import checked,sha,write
from dunhuang.alignment.clock_data import read,ref,observation_contract
from dunhuang.alignment.clock_metrics import grouped,energy_scores
from dunhuang.alignment.service import Reranker
from dunhuang.contracts import matrices
from dunhuang.generation.root_contact import root_contact_features
from dunhuang.generation.diagnostics import upperbody_activity
from dunhuang.evaluation_benchmarks.smpl_mesh import SMPLMesh
from dunhuang.evaluation_benchmarks.physdiff_metrics import population_metrics
from dunhuang.evaluation_benchmarks.edge_pfc import pfc
from dunhuang.evaluation_benchmarks.benchmark_report import official_mesh_parity
from dunhuang.planning.event_plan import semantic_plan_id

def main():
    p=argparse.ArgumentParser()
    for k in ('assets','benchmark-assets','clock','root','diagnosis','replay','plan','query-features','query-receipt','out'):
        p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    assets=read(a.assets);mesh_assets=read(a.benchmark_assets);clock=Path(a.clock);root=Path(a.root)
    replay=read(Path(a.replay)/'result.json')
    assert replay['status']=='FRESH_CLOCK_ROOT_REPLAY_VERIFIED'
    assert replay['clock_protocol_sha256']==sha(clock/'protocol.json')
    assert replay['root_protocol_sha256']==sha(root/'protocol.json')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    generations=read(root/'generation/results.json')
    mesh=SMPLMesh(mesh_assets['mesh_model'],mesh_assets.get('mesh_device','cpu'))
    cache={}
    for r in generations:
        x=np.load(checked(r['motion']));minimum,joints=mesh.sequence(x);cache[(r['arm'],r['control'],r['music_index'])]=(x,minimum,joints)
    # Baseline and ground policy fixed before inspecting generated measurements.
    floors={r['music_index']:dict(value=float(np.percentile(cache[('masked','original',r['music_index'])][1],5)),
        reference_sha256=r['motion']['sha256']) for r in generations if r['arm']=='masked'}
    manifest=read(checked(assets['paired_generation_manifest']));by_uid={r['window_uid']:r for r in manifest['rows']}
    metrics=[]
    for row in generations:
        x,minimum,joints=cache[(row['arm'],row['control'],row['music_index'])]
        folder=Path(row['motion']['path']).parent;physical=read(folder/'physical.json')
        source_row=by_uid[row['source_task']['source_uid']];source=np.load(checked(source_row['motion']))
        available=np.load(checked(source_row['observed']))
        with torch.no_grad():
            actual=root_contact_features(torch.from_numpy(x)[None],torch.ones_like(torch.from_numpy(x)[None],dtype=torch.bool))[0].numpy()
            wanted=root_contact_features(torch.from_numpy(source)[None],torch.from_numpy(available)[None])[0].numpy()
        ar=matrices(x[:,79:127].reshape(-1,8,6));br=matrices(source[:,79:127].reshape(-1,8,6))
        angle=Rotation.from_matrix((ar.swapaxes(-1,-2)@br).reshape(-1,3,3)).magnitude()
        fixed=floors[row['music_index']]
        metrics.append(dict(arm=row['arm'],control=row['control'],music_index=row['music_index'],seed=row['seed'],
            output_sha256=row['motion']['sha256'],legacy=physical['absolute'],native_audit=physical['audit'],
            fixed_mesh_floor_y_m=fixed['value'],fixed_ground_reference_sha256=fixed['reference_sha256'],
            mesh=population_metrics(minimum,joints,fixed['value']),neutral_mesh_pfc=pfc(joints),
            root_displacement_agreement_rmse_m=float(np.sqrt(np.mean((actual[:,:3]-wanted[:,:3])**2))),
            root_velocity_agreement_rmse_mps=float(np.sqrt(np.mean((actual[:,3:6]-wanted[:,3:6])**2))),
            source_upper8_agreement_radians=float(angle.mean()),
            upperbody_activity=upperbody_activity(x),source_upperbody_activity=upperbody_activity(source),
            root_condition_observed_frames=row['root_reference_audit']['observed_root_frames'],
            scope='source agreement diagnostics, not exact cross-capture pose truth; neutral mesh and common ground are proxies'))
    write(out/'generation_metrics.json',metrics)
    first=generations[0];x=np.load(checked(first['motion']))
    parity=official_mesh_parity(mesh,mesh_assets['mesh_model'],x,mesh_assets.get('reference_runtime'))
    write(out/'mesh_parity.json',parity)
    # Frozen old event plan and nonpaired audio, no target pairing/time labels supplied to the service.
    saved_plan=read(a.plan);before=semantic_plan_id(saved_plan);protocol=read(clock/'protocol.json')
    reranker=Reranker(dict({'global':protocol['global_checkpoint']},sync=protocol['sync_checkpoint'],
        qualification=ref('outputs/sync_frozen_01/result.json'),
        music_features=ref(a.query_features),music_receipt=ref(a.query_receipt),transport=ref(clock/'C/head.pt')),
        saved_plan['audio_sha256'])
    trace=[]
    for i,event in enumerate(saved_plan['events'][:2]):
        start,end=event['music_interval'];scores=reranker.scores(event,dict(start=start,end=end),local=True)
        assert not reranker.decision['enabled'] and scores['transport_can_route'] is False
        trace.append(dict(case=i,music_interval=[start,end],scores=scores))
    assert semantic_plan_id(saved_plan)==before
    write(out/'shadow_service.json',dict(status='REAL_NONPAIRED_SHADOW_INFERENCE_VERIFIED',cases=trace,
        decision=reranker.decision,event_plan_unchanged=True,target_ground_truth_used=False,
        no_new_time_warp=True,operational_qualified=False))
    # Paired method deltas preserve query identities and known source clusters.
    rows=read(clock/'data/manifest.json')['rows'];val=[r for r in rows if r['role']=='val']
    wins={}
    for mode in ('A','B','C'):
        with np.load(clock/mode/'measurements.npz') as z:
            score=z['scores'];positive=score[:,2,None];negative=score[:,[0,1,3,4]]
            wins[mode]=((positive>negative)+.5*(positive==negative)).mean(-1)
    with np.load(clock/'data/observations.npz') as z:energy=energy_scores(z)
    ids=[i for i,r in enumerate(rows) if r['role']=='val'];score=energy[ids]
    wins['energy']=((score[:,2,None]>score[:,[0,1,3,4]])+.5*(score[:,2,None]==score[:,[0,1,3,4]])).mean(-1)
    delta={name:grouped(wins[left]-wins[right],val) for name,left,right in [('C_minus_B','C','B'),('C_minus_A','C','A'),('C_minus_energy','C','energy')]}
    write(out/'paired_deltas.json',delta)
    # Export only aggregates; source identities/paths remain in private artifacts.
    clock_result=read(clock/'result.json');root_result=read(root/'result.json')
    stripped={arm:dict(validation=v['validation'],checkpoint_sha256=v['checkpoint']['sha256'],
        frozen_before=v['weights_before'],frozen_after=v['weights_after']) for arm,v in root_result['results'].items()}
    public=dict(schema='clock_root_research_aggregate_v1',analysis_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        clock_code_commit=read(clock/'protocol.json')['code_commit'],root_code_commit=read(root/'protocol.json')['code_commit'],
        diagnosis=read(Path(a.diagnosis)/'summary.json'),alignment=clock_result,
        observation_contract_clarification=observation_contract(dict(
            source='pinned legacy full_audio_observations; combined legacy schema also describes an unused native60 motion path')),
        metadata_erratum='Early manifests copied the combined legacy audio/motion CONTRACT. Actual motion74 is recomputed on decoded native30 recordings, not native60 derivatives. This clarification changes no arrays, weights, scores or population.',
        head_checkpoint_sha256={mode:sha(clock/mode/'head.pt') for mode in ('A','B','C')},
        paired_method_deltas=delta,root_training=dict(results=stripped,initializations=root_result['initializations'],
            steps_per_arm=root_result['training_steps_per_arm']),
        generation=metrics,shadow_service=read(out/'shadow_service.json'),
        mesh_parity={k:v for k,v in parity.items() if k!='model'},fresh_replay=replay,
        seals={name:sha(file) for name,file in dict(clock_protocol=clock/'protocol.json',root_protocol=root/'protocol.json',
            diagnosis_protocol=Path(a.diagnosis)/'protocol.json',replay_result=Path(a.replay)/'result.json',
            generation_metrics=out/'generation_metrics.json',shadow_service=out/'shadow_service.json').items()},
        operational_qualified=False,rootnet_added=False,legacy_limits_changed=False,
        test_evaluations=0,expert_labels_supplied=False,
        limitations=['two exposed Val association components, not a fresh blind evaluation',
            'band-prior-biased clock consistency is not independently labelled action-peak localization',
            'geometric contact is not static-contact truth; common reference ground is not measured scene ground',
            'six short demonstrations and a single seed do not establish full-song quality or generalization'])
    text=json.dumps(public)
    for row in rows:
        for key in ('window_uid','recording_uid','choreography_uid','source_group','music_uid'):
            if row[key] in text:raise ValueError('Raw source identity cannot be published')
    if '/home/' in text or '/storage/' in text:raise ValueError('Private paths cannot be published')
    write(out/'public_summary.json',public)
    write(out/'status.json',dict(status='COMPLETED_READ_ONLY_CLOCK_ROOT_EVALUATION',operational_qualified=False,
        new_generations=0,optimizer_calls=0,shadow_cases=len(trace),generation_cases=len(generations)))
    print('CLOCK_ROOT_EVALUATION_COMPLETED',flush=True)
if __name__=='__main__':main()
