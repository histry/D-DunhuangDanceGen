"""Read-only raw ChangE channels; numerical presence is not world/floor admission."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
def run(manifest,out):
    path=Path(manifest);m=read(path);records=[]
    for r in m['sources']:
        if not r.get('solo_compatible'):continue
        p=path.parent/r['file']
        if sha(p)!=r['sha256']:raise ValueError('Raw source hash mismatch')
        z=np.load(p,allow_pickle=False);poses=z['poses'];trans=z['trans'];fps=float(z['mocap_framerate'])
        if poses.shape[1]!=165 or trans.shape!=(len(poses),3) or fps!=r['source_fps']:raise ValueError('Source state contract')
        tail=poses.reshape(-1,55,3)[:,22:];finite=bool(np.isfinite(trans).all())
        records.append(dict(source_uid=r['source_id'],raw_sha256=sha(p),frames=len(poses),fps=fps,coordinates=r['coordinate_system'],units=r['translation_units'],root_translation_numerically_present=finite,translation_range_m=np.ptp(trans,axis=0).tolist(),root_speed_mean_m_s=float(np.linalg.norm(np.diff(trans,axis=0)*fps,axis=-1).mean()),unobserved_terminal_and_finger_max_abs_axis_angle=float(abs(tail).max()),native24_terminal_truth_available=False,world_floor_support='UNKNOWN',rest_pelvis_and_shape_binding='REQUIRES_SOURCE_SMPLX_BETAS_AND_COORDINATE_VALIDATION',current_body_only_contract_changed=False))
    write(out,dict(manifest_sha256=sha(path),records=records,conclusion='RAW_TRANSLATIONS_EXIST_BUT_HAND_FACE_TAIL_IS_ZERO; NO_AUTOMATIC_NATIVE24_OR_CONTACT_ADMISSION',read_only=True,model_outputs_changed=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.manifest,a.out)
