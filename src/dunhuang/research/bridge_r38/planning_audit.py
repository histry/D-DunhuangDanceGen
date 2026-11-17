"""Read-only content diagnostics. No new path search or semantic completion labels."""
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import Library,read,write,relative_fk,rotation_sample
from dunhuang.research.transition_r37.clock import source_positions
from .view import composition

def run(root):
    from pathlib import Path
    cfg=read('src/dunhuang/research/bridge_r38/config.json');parent=Path(cfg['r37']);lib=Library(read('src/dunhuang/research/complete_r36/config.json')['catalog'])
    z=np.load(parent/'generation/arrays.npz');J=z['skeleton_J22'];parents=z['parents'];result=[]
    for song in read(parent/'planning/result.json')['songs']:
        plan=read(song['new_plan']);report=composition(plan);descriptors=[];details=[]
        for e in plan['events']:
            A,B=e['music_interval'];ticks=np.arange(A,B);q=source_positions(e,ticks);R=rotation_sample(lib.rotations[e['source_uid']],q);P=relative_fk(R,J,parents);speed=np.linalg.norm(np.diff(P,axis=0)*30,axis=-1).mean(-1)
            small=rotation_sample(R,np.linspace(0,len(R)-1,32));descriptors.append(small)
            details.append(dict(event_uid=e.get('event_uid'),source_uid=e['source_uid'],music_interval=[A,B],source_interval=e['source_interval'],body_speed_m_s=float(speed.mean()),low_motion_frame_fraction=float(np.mean(speed<.05)),body_extent_rms_m=float(np.sqrt(np.mean(np.ptp(P,axis=0)**2))),label_scope='KINEMATIC_PROXY_NOT_HOLD_DEVELOPMENT_SEMANTIC_TRUTH'))
        pairs=[]
        for i in range(len(details)):
            for j in range(i+1,len(details)):
                a,b=plan['events'][i],plan['events'][j]
                if a['source_uid']==b['source_uid'] and max(a['source_interval'][0],b['source_interval'][0])<min(a['source_interval'][1],b['source_interval'][1]):continue
                rel=descriptors[i].swapaxes(-1,-2)@descriptors[j];angle=Rotation.from_matrix(rel.reshape(-1,3,3)).magnitude()
                pairs.append(dict(events=[i,j],angle_mean_rad=float(angle.mean()),angle_p95_rad=float(np.percentile(angle,95)),source_intervals_distinct=True))
        music=read(Path(cfg['r36'])/'planning'/(song['id']+'_music.json'));report['acoustic_phrase_proposals']=music['phrases'];report['events']=details;report['closest_distinct_interval_pairs']=sorted(pairs,key=lambda x:x['angle_mean_rad'])[:10]
        result.append(dict(song=song['id'],**report))
    write(Path(root)/'planning_diagnostics.json',dict(songs=result,plans_unchanged=True,mechanical_vs_theme_repetition='NOT_SEMANTICALLY_IDENTIFIED',source_balancing_enforced=False,sdmr_used=False))
