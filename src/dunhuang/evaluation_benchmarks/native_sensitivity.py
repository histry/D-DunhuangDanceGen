"""Fixed-plane, skeleton, duration and noise measurements on immutable motions."""
from pathlib import Path
from collections import defaultdict
import numpy as np
from dunhuang.io import checked,write,sha
from .smpl_mesh import SMPLMesh
from .benchmark_report import official_mesh_parity
from .physdiff_metrics import population_metrics as metrics
from .contact_evidence import conditioned_metrics
from .calibration import jerk_report
from .protocols import GroundAndSkeletonContract
from dunhuang.data.conversion import decode


def sensitivity(backend,selected,generated,mesh_assets,config,out):
    import torch
    out=Path(out);out.mkdir();(out/'geometry').mkdir()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    mesh=SMPLMesh(mesh_assets['mesh_model'],mesh_assets.get('mesh_device','cpu'))
    first=np.load(checked(selected[0]['motion']))
    write(out/'mesh_reference_parity.json',official_mesh_parity(mesh,mesh_assets['mesh_model'],first,mesh_assets.get('reference_runtime')))
    cache={}
    def geometry(x,key):
        if key not in cache:
            minimum,joints=mesh.sequence(x)
            file=out/'geometry'/(key+'.npz');np.savez_compressed(file,minimum_y=minimum,joints=joints)
            cache[key]=(minimum,joints,dict(path=str(file),sha256=sha(file)))
        return cache[key]
    rows=[];floors={}
    for g in generated:
        # The generated evaluator froze a baseline Native ground reference for this cohort.
        anchor=g['anchor']['motion'];x=np.load(checked(anchor));m,j,c=geometry(x,anchor['sha256'])
        floors[anchor['sha256']]=dict(native_foot_floor=g['anchor']['value'],
            common_mesh_floor=float(np.percentile(m,5)),reference=anchor,mesh_cache=c)
    write(out/'frozen_cohort_ground.json',floors)
    for g in generated:
        x=np.load(checked(g['motion']));native=np.load(checked(g['fk']))
        minimum,joints,cache_ref=geometry(x,g['motion']['sha256'])
        anchor=floors[g['anchor']['motion']['sha256']]
        candidate_floor=float(np.percentile(native[:,[7,8,10,11],1],5))
        tests=[('frozen_common_mesh',anchor['common_mesh_floor']),('frozen_common_native_joint',anchor['native_foot_floor']),
               ('estimated_candidate_native_joint',candidate_floor)]
        for name,floor in tests:
            row=dict(case=g['name'],mode=g['mode'],domain=g['domain'],protocol=name,floor_y_m=floor,
                floor_ground_truth=False,motion=g['motion'],mesh_cache=cache_ref,
                joint_geometry_max_abs_m=float(abs(joints-native).max()),
                foot_geometry_max_abs_m=float(abs(joints[:,[7,8,10,11]]-native[:,[7,8,10,11]]).max()),
                contract=GroundAndSkeletonContract('neutral SMPL beta0',mesh_assets['mesh_model']['sha256'],floor,
                    'estimated_candidate_floor' if name.startswith('estimated') else 'frozen_common_reference',
                    g['motion']['sha256'] if name.startswith('estimated') else anchor['reference']['sha256'],len(x)).record(),
                neutral_mesh=metrics(minimum,joints,floor,contact_height=.05),
                native_joint_height_min_m=float((native[:,[7,8,10,11],1]-floor).min()),
                neutral_joint_height_min_m=float((joints[:,[7,8,10,11],1]-floor).min()),
                physical_depth_is_mesh_only=True,legacy_result=g['analysis']['legacy']['ok'])
            rows.append(row)
        write(out/'generated_progress.json',dict(completed=len(rows)//3,total=len(generated)))
    write(out/'generated_ground_skeleton.json',rows)
    source=[]
    for r in selected:
        raw,observed=decode(checked(r['raw_motion']),'aistpp')
        start=r['start_frame'];reference=raw[start:start+150].copy()
        ref_sha=sha(checked(r['raw_motion']));minimum,mesh_j,cache_ref=geometry(reference,r['motion']['sha256']+'_raw')
        common=float(np.percentile(minimum,5))
        for length in config['sensitivity']['length_frames']:
            if start+length>len(raw):
                source.append(dict(window_uid=r['window_uid'],length_frames=length,status='INSUFFICIENT_REAL_FRAMES_NO_PADDING'))
                continue
            x=raw[start:start+length].copy();fk=backend.fk(x)
            source.append(dict(window_uid=r['window_uid'],recording_uid=r['recording_uid'],length_frames=length,
                status='COMPUTED',source_motion=r['raw_motion'],source_start_frame=start,
                jerk=jerk_report(fk,backend.limits['joint_jerk_mps3_max']),
                legacy=backend.gate(backend.audit(x,backend.cfg,precomputed_joints=fk)),
                same_raw_prefix=True,global_frame_source='real decoded frames; no concatenation, looping or padding'))
        # Three fixed plane values for each skeleton on the identical150 frames.
        fk=backend.fk(reference)
        for offset in config['sensitivity']['floor_offsets_m']:
            floor=common+offset
            source.append(dict(window_uid=r['window_uid'],status='GROUND_SKELETON_SENSITIVITY',floor_offset_m=offset,
                floor_y_m=floor,reference_geometry='first raw150 neutral mesh minimum-y p05, unvalidated proxy',
                native_height_min_m=float((fk[:,[7,8,10,11],1]-floor).min()),
                neutral_height_min_m=float((mesh_j[:,[7,8,10,11],1]-floor).min()),
                geometry_max_abs_m=float(abs(mesh_j-fk).max()),
                native_observation=conditioned_metrics(fk,floor,low=backend.low_filter(fk[:,[7,8,10,11],1]<=floor+.055)),
                neutral_observation=conditioned_metrics(mesh_j,floor,low=backend.low_filter(mesh_j[:,[7,8,10,11],1]<=floor+.055)),
                mesh=metrics(minimum,mesh_j,floor,contact_height=.05)))
        for amplitude in config['sensitivity']['measurement_jitter_amplitudes_m']:
            x=reference.copy();x[:,5]+=amplitude*np.sin(2*np.pi*config['sensitivity']['jitter_hz']*np.arange(150)/30)
            perturbed=backend.fk(x)
            source.append(dict(window_uid=r['window_uid'],status='NOISE_SENSITIVITY',jitter_amplitude_m=amplitude,
                frequency_hz=config['sensitivity']['jitter_hz'],operator='RootY sinusoid; in-memory copy only',
                jerk=jerk_report(perturbed,backend.limits['joint_jerk_mps3_max']),
                legacy=backend.gate(backend.audit(x,backend.cfg,precomputed_joints=perturbed))))
    write(out/'source_duration_ground_noise.json',source)
    long_results=[]
    for folder in config['sensitivity']['saved_long_cases']:
        folder=Path(folder);x=np.load(folder/'native.npy');fk=np.load(folder/'native_fk.npy')
        original=__import__('json').loads((folder/'physical.json').read_text())
        actual=backend.audit(x,backend.cfg,precomputed_joints=fk)
        if actual!=original['audit'] or backend.gate(actual)!=original['absolute']:
            raise ValueError('Long saved-motion Legacy replay changed')
        for length in config['sensitivity']['saved_prefix_frames']:
            if length>len(x):continue
            long_results.append(dict(run=str(folder),motion_sha256=sha(folder/'native.npy'),
                prefix_frames=length,same_saved_motion_prefix=True,
                jerk=jerk_report(fk[:length],backend.limits['joint_jerk_mps3_max']),
                legacy=backend.gate(backend.audit(x[:length],backend.cfg,precomputed_joints=fk[:length]))))
    write(out/'saved_long_prefixes.json',long_results)
    write(out/'status.json',dict(status='COMPLETED_READ_ONLY_SENSITIVITY',generated_cases=len(generated),
        source_recordings=len(selected),mesh_model=mesh.description,ground_truth=False,
        skeletons_not_interchangeable=True,physdiff_author_code_parity=False,motion_updates=0,optimizer_calls=0))
