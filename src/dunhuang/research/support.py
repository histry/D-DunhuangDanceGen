"""Exhaustive channel/window support and coordinate diagnosis; no repair."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,write
from dunhuang.contracts import MotionAdapter,matrices
from .frozen import payload

def label(i):
    if i<4:return 'predicted_contact_'+str(i)
    if i<7:return 'EDGE_root_'+('x','y','z')[i-4]
    return 'joint_%02d_row6d_%d'%((i-7)//6,(i-7)%6)

def audit(assets,out):
    checkpoint=payload(assets);scaler=checkpoint['normalizer'].scaler
    scale=scaler.scale_.cpu().numpy();bias=scaler.min_.cpu().numpy()
    training_min=scaler.data_min_.cpu().numpy();training_max=scaler.data_max_.cpu().numpy()
    with np.load(checked(assets['scaler'])) as exported:
        if not np.array_equal(scale,exported['scale']) or not np.array_equal(bias,exported['bias']):raise ValueError('Scaler binding differs')
    del checkpoint
    catalog=json.loads(checked(assets['catalog']).read_text())
    refs={r['source_uid']:(r.get('recording_uid',r['source_uid']),r['reference']) for r in catalog['entries']}
    window_masks=[];window_ids=[];source_rows=[]
    min_raw=np.full(151,np.inf);max_raw=-min_raw.copy();lower_count=np.zeros(151,int);upper_count=np.zeros(151,int)
    excess=np.zeros(151);global_frames=0;normalized_min=np.full(151,np.inf);normalized_max=-normalized_min.copy()
    hypotheses={k:0 for k in ('correct_first_XZ_origin','no_root_origin','wrong_row_assumption')}
    for uid,(group,ref) in refs.items():
        source=np.load(checked(ref),allow_pickle=False)
        edge=MotionAdapter.to_edge(source.copy())
        native=MotionAdapter.to_native(edge.copy())
        r=matrices(source[:,7:].reshape(-1,24,6));rr=matrices(native[:,7:].reshape(-1,24,6))
        angles=Rotation.from_matrix((r.swapaxes(-1,-2)@rr).reshape(-1,3,3)).magnitude()
        source_rows.append(dict(source_uid=uid,recording_group=group,reference=ref,frames=len(source),
            root_roundtrip_max_abs_m=float(abs(native[:,4:7]-source[:,4:7]).max()),
            local_rotation_roundtrip_max_rad=float(angles.max()),
            predicted_contacts_preserved=bool(np.array_equal(source[:,:4],native[:,:4])),
            raw_first_vector_norm_error=float(abs(np.linalg.norm(source[:,7:].reshape(-1,24,6)[:,:,:3],axis=-1)-1).max()),
            coordinate_scope='decoded canonical Y-up columns -> EDGE Z-up rows; no new raw-camera calibration claim'))
        for start in range(0,len(source)-149,75):
            x=source[start:start+150].copy();origin=x[0,4:7].copy();origin[1]=0.
            correct=MotionAdapter.to_edge(x.copy(),origin)
            z=correct*scale+bias;tol=32*np.finfo(np.float32).eps
            supported=abs(z)<=1+tol
            window_masks.append(supported.all(0));window_ids.append(dict(source_uid=uid,start=start))
            min_raw=np.minimum(min_raw,correct.min(0));max_raw=np.maximum(max_raw,correct.max(0))
            normalized_min=np.minimum(normalized_min,z.min(0));normalized_max=np.maximum(normalized_max,z.max(0))
            lower_count+=(z<-1-tol).sum(0);upper_count+=(z>1+tol).sum(0)
            excess=np.maximum(excess,np.maximum(abs(z)-1,0).max(0));global_frames+=len(x)
            hypotheses['correct_first_XZ_origin']+=int(supported.all())
            hypotheses['no_root_origin']+=int((abs(MotionAdapter.to_edge(x.copy())*scale+bias)<=1+tol).all())
            wrong=correct.copy()
            # Deliberately wrong layout is only a negative-control diagnosis.
            wrong[:,7:]=x[:,7:]
            hypotheses['wrong_row_assumption']+=int((abs(wrong*scale+bias)<=1+tol).all())
    masks=np.array(window_masks);np.savez_compressed(out/'support_windows.npz',channel_supported=masks)
    channels=[]
    for i in range(151):
        channels.append(dict(index=i,name=label(i),training_data_min=float(training_min[i]),training_data_max=float(training_max[i]),
            training_near_constant=bool(training_max[i]-training_min[i]<10*np.finfo(np.float32).eps),
            physical_box_min=float((-1-bias[i])/scale[i]),physical_box_max=float((1-bias[i])/scale[i]),
            source_min=float(min_raw[i]),source_max=float(max_raw[i]),
            normalized_source_min=float(normalized_min[i]),normalized_source_max=float(normalized_max[i]),
            lower_violating_joint_frames=int(lower_count[i]),upper_violating_joint_frames=int(upper_count[i]),
            max_normalized_excess=float(excess[i]),supported_windows=int(masks[:,i].sum())))
    groups={name:int(masks[:,a:b].all(1).sum()) for name,(a,b) in dict(contact=(0,4),root=(4,7),body8=(79,127),wrists=(127,139),terminal_hands=(139,151),full151=(0,151)).items()}
    result=dict(schema='full151_support_diagnosis_v1',windows=len(masks),window_weighted_frames=global_frames,
        group_supported_windows=groups,hypothesis_full_supported_windows=hypotheses,channels=channels,
        sources=source_rows,window_ids=window_ids,normalizer_refitted=False,targets_clipped=False,
        bounds='exact frozen affine output box [-1,1], distinct from semantic/observed-target admission',
        caveat='contact is the canonical source prediction, not independently authenticated contact ground truth')
    write(out/'support.json',result)
    return result
