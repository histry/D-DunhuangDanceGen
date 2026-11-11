"""Verify saved B3 ownership, physical metrics and bounded edits without new generation."""
import sys
from pathlib import Path
import numpy as np
from b2 import ROOT,read,write,sha,matrices,angle_difference

def verify(folder,destination):
    parent=ROOT/'outputs/b2_whole_f06bc86/whole_temporal'
    base=np.load(parent/'native.npy')
    positions=np.load(parent/'native_fk.npy')
    original=read(parent/'physical.json')
    summary=read(folder/'summary.json')
    rows=summary['candidates']+([summary['continuation']] if summary['continuation'] else [])
    receipts=[]
    for row in rows:
        case=folder/row['case']
        motion=np.load(case/'native.npy');fk=np.load(case/'native_fk.npy')
        audit=read(case/'physical.json')
        a,b=row['span'];allowed=np.zeros_like(base,dtype=bool)
        allowed[a+1:b-1,4:7]=True
        for joint in row['joints']:allowed[a+1:b-1,7+6*joint:13+6*joint]=True
        if motion[~allowed].tobytes()!=base[~allowed].tobytes():raise ValueError('Unowned bytes changed')
        angle=angle_difference(matrices(base[:,7:].reshape(-1,24,6)).astype(float),
                               matrices(motion[:,7:].reshape(-1,24,6)).astype(float))
        rotation=float(angle.max())
        translation=float(np.linalg.norm(motion[:,4:7]-base[:,4:7],axis=-1).max())
        if rotation>.1*row['amount']+1e-5 or translation>.02*row['amount']+1e-6:
            raise ValueError('Trust region escaped')
        parents=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21])
        lengths=np.linalg.norm(fk[:,1:]-fk[:,parents[1:]],axis=-1)
        old_lengths=np.linalg.norm(positions[:,1:]-positions[:,parents[1:]],axis=-1)
        bone_error=float(abs(lengths-old_lengths).max())
        if bone_error>1e-5:raise ValueError('Bone lengths changed')
        jerk=float(np.linalg.norm(np.diff(fk,n=3,axis=0),axis=-1).max()*27000)
        if not np.isclose(jerk,audit['audit']['joint_jerk_mps3_max'],rtol=1e-5):raise ValueError('FK D3 mismatch')
        if original['absolute']['limits']!=audit['absolute']['limits']:raise ValueError('Native threshold mismatch')
        receipts.append(dict(case=row['case'],motion_sha256=sha(case/'native.npy'),rotation_max_rad=rotation,
            translation_max_m=translation,bone_length_max_difference_m=bone_error,owned_bytes_verified=True))
    result=dict(provenance=read(folder/'provenance.json'),attribution=read(folder/'attribution.json'),
        root_only=read(folder/'root_only.json'),primitives=read(folder/'primitives.json'),
        joint_selection=read(folder/'joint_selection.json'),responses=summary,verification=receipts,
        baseline_metrics={key:original['audit'][key] for key in original['absolute']['limits'] if key in original['audit']},
        baseline_reasons=original['absolute']['reasons'],scope='Single frozen B2 temporal song, no neural forwards')
    write(destination,result)
    print('B3_VERIFIED',len(receipts))
if __name__=='__main__':verify(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
