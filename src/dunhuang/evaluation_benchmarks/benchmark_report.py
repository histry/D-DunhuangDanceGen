"""Immutable replay of saved motions with independent paper benchmark reports."""
import contextlib,importlib.util,io,json,pickle,sys,tempfile,subprocess
from pathlib import Path
import numpy as np
from dunhuang.io import checked,sha,write,pinned
from dunhuang.contracts import Q,matrices
from dunhuang.generation.diagnostics import upperbody_activity
from .edge_pfc import pfc
from .physdiff_metrics import metrics
from .contact_states import diagnose
from .boundary_metrics import boundaries
from .smpl_mesh import SMPLMesh,load_model

def read(p):return json.loads(Path(p).read_text())

def official_mesh_parity(mesh,model_ref,motion,runtime):
    import torch,importlib.metadata
    if runtime:sys.path.insert(0,str(Path(runtime).resolve()))
    from smplx.lbs import lbs
    version=importlib.metadata.version('smplx')
    if version!='0.1.28':raise ValueError('Pinned SMPL reference runtime differs')
    data=load_model(model_ref)
    jreg=data['J_regressor'].toarray()
    x=motion[np.unique(np.linspace(0,len(motion)-1,min(4,len(motion))).astype(int))]
    r=torch.as_tensor(matrices(x[:,7:].reshape(-1,24,6)),dtype=torch.float32,device=mesh.device)
    with torch.no_grad():
        v,j=lbs(torch.zeros(len(x),10,device=mesh.device),r,mesh.template,
            torch.as_tensor(data['shapedirs'],dtype=torch.float32,device=mesh.device),
            mesh.posedirs.T.contiguous(),torch.as_tensor(jreg,dtype=torch.float32,device=mesh.device),
            torch.tensor(mesh.parents,device=mesh.device,dtype=torch.long),mesh.weights,pose2rot=False)
        translation=torch.as_tensor(x[:,4:7],device=mesh.device)-mesh.rest[0]
        v+=translation[:,None];j+=translation[:,None]
        ours_v,ours_j=mesh.forward(x)
        error_v=float(abs(v-ours_v).max());error_j=float(abs(j-ours_j).max())
    if max(error_v,error_j)>1e-5:raise ValueError('Canonical SMPL mesh differs from smplx reference')
    return dict(runtime='smplx=='+version,vertices_max_abs_m=error_v,joints_max_abs_m=error_j,tolerance_m=1e-5,
        source='https://github.com/vchoutas/smplx',model=model_ref,shape='shared neutral beta0')

def report(assets,manifest,out):
    import torch
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    document=read(manifest);code_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[3]),'rev-parse','HEAD'],text=True).strip()
    edge=pinned(assets['edge_root'],assets['edge_commit']);oracle_file=edge/'eval/eval_pfc.py'
    spec=importlib.util.spec_from_file_location('edge_pfc_official',oracle_file)
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    mesh=SMPLMesh(assets['mesh_model'],device=assets.get('mesh_device','cpu'))
    if document.get('floor_policy')!='frozen_baseline_mesh_p05':raise ValueError('Explicit frozen common-floor policy required')
    cache={};records=[];input_hashes={}
    def geometry(folder):
        folder=Path(folder).resolve();native=folder/'native.npy';fk=folder/'native_fk.npy'
        key=sha(native);input_hashes[str(native)]=key;input_hashes[str(fk)]=sha(fk)
        for name in ('physical.json','request.json','generation.json'):
            if (folder/name).exists():input_hashes[str(folder/name)]=sha(folder/name)
        if key not in cache:
            motion=np.load(native,allow_pickle=False);minimum,joints=mesh.sequence(motion)
            file=out/(key[:16]+'.mesh.npz');np.savez_compressed(file,minimum_y=minimum,joints=joints)
            cache[key]=(motion,minimum,joints,file)
        return cache[key],np.load(fk,allow_pickle=False),key
    first=Path(document['cohorts'][0]['baseline']).resolve()
    first_motion=np.load(first/'native.npy')
    parity=official_mesh_parity(mesh,assets['mesh_model'],first_motion,assets.get('reference_runtime'))
    write(out/'mesh_reference_parity.json',parity)
    write(out/'protocol.json',dict(schema='dance_paper_benchmarks_v1',code_commit=code_commit,
        manifest_sha256=sha(manifest),assets_sha256=sha(assets['_file']) if assets.get('_file') else None,
        edge_commit=assets['edge_commit'],edge_pfc_source_sha256=sha(oracle_file),mesh=mesh.description,
        floor_policy=document['floor_policy'],floor_ground_truth=False,
        floor_note='one frozen baseline mesh 5th-percentile minimum per cohort; never fit ground separately to a candidate',
        physdiff='Appendix A reimplementation; author evaluator unavailable; no direct leaderboard comparability',
        contact_height_m=document['contact_height_m'],geometry_tolerance_m=.005,
        native_acceptance='historical frozen result, not recomputed or overwritten',motion_updates=0,optimizer_steps=0))
    for cohort in document['cohorts']:
        (base,base_minimum,_,_),_,_=geometry(cohort['baseline'])
        floor=float(np.percentile(base_minimum,5))
        for item in [dict(name=cohort['name']+'/baseline',run=cohort['baseline'])]+cohort['cases']:
            folder=Path(item['run']).resolve()
            (motion,minimum,joints,mesh_file),native_fk,motion_sha=geometry(folder)
            if len(motion)!=len(native_fk):raise ValueError('Saved FK clock mismatch')
            native=read(folder/'physical.json');request=read(folder/'request.json')
            score=pfc(native_fk)
            with tempfile.TemporaryDirectory(prefix='pfc_oracle_') as temp:
                with (Path(temp)/'motion.pkl').open('wb') as f:pickle.dump(dict(full_pose=native_fk@Q.T),f)
                output=io.StringIO()
                with contextlib.redirect_stdout(output),contextlib.redirect_stderr(io.StringIO()):
                    oracle.calc_physical_score(temp)
                official=float(output.getvalue().strip().split()[-1])
            if score['value'] is None:
                if np.isfinite(official):raise ValueError('Undefined PFC differs from official implementation')
            elif not np.isclose(score['value'],official,rtol=1e-12,atol=1e-12):
                raise ValueError('PFC differs from unchanged official evaluator')
            plan=read(folder/'event_plan.json') if (folder/'event_plan.json').exists() else None
            event_edges=[e['music_interval'][0] for e in plan['events'][1:]] if plan else []
            record=dict(name=item['name'],run=str(folder),frames=len(motion),seed=request['seed'],cohort=cohort['name'],
                motion_sha256=motion_sha,fk_sha256=sha(folder/'native_fk.npy'),mesh_cache=dict(path=str(mesh_file),sha256=sha(mesh_file)),
                pfc=dict(score,official_value=float(official) if np.isfinite(official) else None,official_parity=True),
                physdiff=metrics(minimum,joints,floor,contact_height=document['contact_height_m']),
                contact_extension=diagnose(joints,floor,motion[:,:4],contact_height=document['contact_height_m']),
                floor_reference_run=cohort['baseline'],floor_y_m=floor,
                floor_reference_seed=read(Path(cohort['baseline'])/'request.json')['seed'],
                native_mesh_joints_max_abs_m=float(abs(joints-native_fk).max()),
                historical_native=native['absolute'],historical_native_limits=native['limits'],
                historical_jerk_max_mps3=native['audit']['joint_jerk_mps3_max'],
                historical_native_skate_p95_mps=native['audit']['foot_skate_mps_p95'],
                upperbody_activity=upperbody_activity(motion),
                event_boundaries=boundaries(native_fk,event_edges),
                window_boundaries=boundaries(native_fk,range(75,len(motion)-74,75)),
                legacy_music_diagnostic=read(folder/'evaluation.json'),
                independent_style='NOT_EVALUATED',human_naturalness='NOT_EVALUATED')
            records.append(record);print('BENCHMARK',record['name'],score['value'],record['physdiff']['phys_err_mm'],flush=True)
            write(out/'progress.json',dict(completed=len(records),last=record['name']))
    for file,digest in input_hashes.items():
        if sha(file)!=digest:raise ValueError('Read-only benchmark changed input '+file)
    result=dict(schema='benchmark_and_acceptance_v1',records=records,mesh_reference_parity=parity,
        unique_motions=len(cache),cases=len(records),official_pfc_parity_all=True,inputs_unchanged=True,
        physdiff_author_code_parity=False,contact_semantic_calibration='BLOCKED_NO_VERIFIED_REAL_LABELS',
        input_hashes=input_hashes,motion_updates=0,optimizer_steps=0)
    write(out/'report.json',result)
    write(out/'status.json',dict(status='COMPLETED_READ_ONLY_BENCHMARK',cases=len(records),
        official_pfc_parity_all=True,mesh_backend_parity=True,native_results_reclassified=False,motion_updates=0))
    return result
