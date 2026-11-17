"""Recover source-fitted relative translation derivatives without inventing shaped world FK."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,sha,Library

def translation_channels(trans,fps):
    x=np.asarray(trans,float)
    if x.ndim!=2 or x.shape[1]!=3 or len(x)<4 or not np.isfinite(x).all() or fps<=0:raise ValueError('Finite measured translation and actual fps')
    return dict(displacement=x-x[:1],velocity=np.diff(x,axis=0)*fps,acceleration=np.diff(x,n=2,axis=0)*fps**2)

def run(manifest,out):
    path=Path(manifest);m=read(path);lib=Library(read('src/dunhuang/research/complete_r36/config.json')['catalog']);source={r['source_id']:r for r in m['sources']};records=[];arrays={}
    for uid in sorted(lib.rotations):
        r=source[uid];p=path.parent/r['file']
        if sha(p)!=r['sha256']:raise ValueError('Original source identity changed')
        z=np.load(p,allow_pickle=False);poses=z['poses'].reshape(-1,55,3);trans=z['trans'];fps=float(z['mocap_framerate'])
        if len(trans)!=len(lib.rotations[uid]) or fps!=30 or r['coordinate_system']!='y_up' or r['translation_units']!='m':raise ValueError('Unverified source/shared clock or axes')
        rawR=Rotation.from_rotvec(poses[:,:22].reshape(-1,3)).as_matrix().reshape(-1,22,3,3);error=Rotation.from_matrix((rawR.swapaxes(-1,-2)@lib.rotations[uid]).reshape(-1,3,3)).magnitude()
        if error.max()>1e-5:raise ValueError('Source-to-common body axes not established')
        values=translation_channels(trans,fps)
        for k,v in values.items():arrays[uid+'_'+k]=v
        native=lib.native[uid][:,4:7];offset=np.median(native-trans,axis=0);residual=native-trans-offset
        records.append(dict(source_uid=uid,raw_sha256=sha(p),source_fps=fps,frames=len(trans),gender=str(z['gender']),betas_sha256=__import__('hashlib').sha256(z['betas'].tobytes()).hexdigest(),body_rotation_parity_max_rad=float(error.max()),native_translation_offset_m=offset.tolist(),native_translation_constant_offset_max_error_m=float(abs(residual).max()),relative_translation_derivatives_supported=True,scope='SOURCE_SMPLX_FIT_Y_UP_METERS; RELATIVE_TRANSLATION_ONLY_NOT_WORLD_FOOT_TRUTH',absolute_pelvis_world_gauge='UNKNOWN_REQUIRES_SHAPED_J0',source_shaped_world_fk='NOT_ADMITTED_SOURCE_GENDER_MODEL_NOT_LOCATED',ground_and_contact='UNKNOWN',native24='NOT_EVALUABLE_22_OF24',missing_tail_max_abs=float(abs(poses[:,22:]).max())))
    out=Path(out);out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'arrays.npz',**arrays);result=dict(records=records,manifest_sha256=sha(path),source_model_inventory_scope='REGISTERED_PROJECTS_AND_STORAGE_PATHS; NO_GENDERED_SMPLX_TEMPLATE_LOCATED',root_derivatives_can_be_used_in_separately_registered_training=True,source_world_fk_admitted=False,outputs_from_main_experiment_changed=False,optimizer_steps=0)
    write(out/'result.json',result);print('R39_SOURCE_ROOT_DERIVATIVES',len(records),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.manifest,a.out)
