"""Alternate-source endpoint graph. Different identities, never old reconstruction truth."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,Library,sha
from dunhuang.research.bridge_audit_r46.admission import certificate
from .geometry import segmented,endpoint_jets

def run(index,out):
    cfg=read(Path(__file__).with_name('config.json'));meta=read(Path(index)/'result.json');lib=Library(read('src/dunhuang/research/complete_r36/config.json')['catalog']);nodes=[];edges=[]
    for uid in sorted(lib.rows):
        options=sorted([r for r in meta['units'] if r['source_uid']==uid and r['duration_frames']>=6],key=lambda r:r['source_interval']);nodes.append(options[len(options)//2])
    for left in nodes:
        for right in nodes:
            if left['source_uid']==right['source_uid']:continue
            a=left['source_interval'][1];b=right['source_interval'][0];R=np.broadcast_to(np.eye(3),(256,22,3,3)).copy();R[101:104]=lib.rotations[left['source_uid']][a-3:a];R[152:155]=lib.rotations[right['source_uid']][b:b+3];R[104:152]=np.nan;candidate,c=segmented(R,cfg['gap'],cfg['segment_counts'],cfg['chart_radius_rad']);old=certificate(R,cfg['gap']);angles=Rotation.from_matrix(R[103].swapaxes(-1,-2)@R[152]).magnitude();jet=np.mean([np.linalg.norm(endpoint_jets(R[:,j],cfg['gap'])[0]-endpoint_jets(R[:,j],cfg['gap'])[2]) for j in range(22)]);cost=float(angles.mean()+.1*jet);edges.append(dict(left_unit_uid=left['unit_uid'],right_unit_uid=right['unit_uid'],left_source_uid=left['source_uid'],right_source_uid=right['source_uid'],left_observed_interval=[a-3,a],right_observed_interval=[b,b+3],mean_endpoint_angle_rad=float(angles.mean()),mean_velocity_mismatch_rad_per_frame=float(jet),source_only_cost=cost,G0=old,G1=c,eligible=candidate is not None,hidden_truth='NO_UNIQUE_CROSS_SOURCE_TRUTH',world_root='UNKNOWN_NOT_PLANNED',contact='UNKNOWN',semantic_fit='NOT_EVALUABLE',can_route=False))
    chosen=[]
    for n in nodes:
        options=[e for e in edges if e['left_unit_uid']==n['unit_uid'] and e['eligible']];chosen.append(dict(left_unit_uid=n['unit_uid'],choice=min(options,key=lambda e:(e['source_only_cost'],e['right_unit_uid']))['right_unit_uid'] if options else None))
    write(out,dict(nodes=nodes,edges=edges,choices=chosen,node_selection='METADATA_MIDDLE_UNIT_PER_SOURCE_NOT_MAX_ACTIVITY',index_sha256=sha(Path(index)/'result.json'),comparison_scope='G3_ALTERNATIVE_SOURCE_PAIRING_NOT_SAME_ENDPOINT_RECONSTRUCTION',training_steps=0,can_route=False,operational_qualified=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.index,a.out)
