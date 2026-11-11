"""Joint selection on qualified event representatives, with explicit infeasibility scope."""
import itertools
import sys
from pathlib import Path
import numpy as np
from b2 import ROOT,read,pinned,sha,write,MotionAdapter,matrices,angle_difference

def main(folder):
    q=read(ROOT/'outputs/b2_whole_f06bc86/whole_temporal/request.json')
    sys.path.insert(0,str(pinned(q['legacy_root'],q['legacy_commit'])))
    from motion.retrieval_generation import decode_continuous_source
    root=Path(q['asset_root'])
    catalog=read(root/'outputs/unpaired_data_protocol_20261015_ZY1C32/catalog/motion_catalog.json')
    scaler=np.load(ROOT/'outputs/b2_c7be67c/5s_custom_none/scaler.npz')
    index=read(folder/'primitives.json')
    references=[]
    for row in index['groups']:
        source=next(s for s in catalog['source_records'] if s['source_uid']==row['source_uid'])
        raw=root/'assets/motion/smpl_official_14'/Path(source['raw_receipt']['path']).name
        if sha(raw)!=source['source_sha256']:raise ValueError('Source identity changed')
        t=row['representative_center']
        clip,_=decode_continuous_source(raw,source['discovery_metadata'],(t-12,t+13),known_rows=np.ones(25,bool))
        v=MotionAdapter.to_edge(clip)[:,79:139]
        if np.any(abs(v*scaler['scale'][79:139]+scaler['bias'][79:139])>1+32*np.finfo(np.float32).eps):
            raise ValueError('Representative support recheck failed')
        r=matrices(clip[:,79:139].reshape(25,10,6)).astype(float)
        references.append(dict(entry=r[0],exit=r[-1],activity=float(angle_difference(r[:-1],r[1:]).mean()*30)))
    slots=read(root/'outputs/edge_baseline_A65_01/inputs/music_mssd.json')['slots']
    densities=[next(s['target_motion_density'] for s in slots if s['music_start']<=a['frame']<s['music_end']) for a in q['anchors']]
    activity=np.array([r['activity'] for r in references])
    feasible=[]
    for order in itertools.permutations(range(len(references)),3):
        if len({index['groups'][i]['source_uid'] for i in order})!=3:continue
        gaps=[float(angle_difference(references[a]['exit'],references[b]['entry']).mean()) for a,b in zip(order,order[1:])]
        if max(gaps)>.5:continue
        score=sum(abs(float(np.mean(activity<activity[i]))-d) for i,d in zip(order,densities))+.25*sum(gaps)
        feasible.append(dict(order=list(order),mean_joint_gaps_rad=gaps,score=score))
    feasible.sort(key=lambda row:(row['score'],row['order']))
    write(folder/'joint_selection.json',dict(independent_events=index['event_groups'],sources=index['source_count'],
        representative_count=len(references),compatible_three_source_sequences=len(feasible),selected=feasible[:1],
        music_frames=[a['frame'] for a in q['anchors']],music_densities=densities,
        source_duration_frames=25,target_duration_frames=25,all_representatives_support_rechecked=True,
        compatibility_cap_mean_rad=.5,
        status='FEASIBLE_REPRESENTATIVE_SEQUENCE' if feasible else 'NO_COMPATIBLE_THREE_SOURCE_SEQUENCE_IN_REPRESENTATIVES',
        scope='One verified representative per event. No claim about all possible source trajectories.'))
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
