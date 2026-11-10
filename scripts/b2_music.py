"""One five-second follow-up using in-support ten-joint trajectories and frozen music slots."""
import json
import sys
from pathlib import Path
import numpy as np
from b2 import ROOT,read,checked,sha,write,pinned,MotionAdapter,matrices,angle_difference
from dunhuang.pipeline import run
from dunhuang.retrieval import select_supported_phrase

def main():
    b2=Path(sys.argv[1]).resolve()
    if not read(b2/'5s_comparison.json')['advance']:
        raise ValueError('Five-second temporal comparison did not qualify for follow-up')
    out=Path(sys.argv[2]).resolve()
    out.mkdir(parents=True,exist_ok=False)
    q=read(b2/'5s_temporal'/'request.json')
    sys.path.insert(0,str(pinned(q['legacy_root'],q['legacy_commit'])))
    from motion.retrieval_generation import decode_continuous_source
    root=Path(q['asset_root'])
    catalog_path=root/'outputs/unpaired_data_protocol_20261015_ZY1C32/catalog/motion_catalog.json'
    catalog=read(catalog_path)
    slots_path=root/'outputs/edge_baseline_A65_01/inputs/music_mssd.json'
    slots=read(slots_path)
    if sha(q['audio']['path'])!=q['audio']['sha256'] or sha(root/'assets/music/test/audio'/Path(slots['audio']).name)!=q['audio']['sha256']:
        raise ValueError('Music slot binding differs from fixed query')
    scaler=np.load(b2/'5s_custom_none/scaler.npz')
    candidates=[]
    for source in catalog['source_records']:
        raw=checked(dict(path=str(root/'assets/motion/smpl_official_14'/Path(source['raw_receipt']['path']).name),sha256=source['source_sha256']))
        n=source['raw_frames']
        motion,_=decode_continuous_source(raw,source['discovery_metadata'],(0,n),known_rows=np.ones(n,bool))
        edge=MotionAdapter.to_edge(motion)
        valid=(abs(edge[:,79:139]*scaler['scale'][79:139]+scaler['bias'][79:139])<=1+32*np.finfo(np.float32).eps).all(1)
        for event in catalog['events']:
            if event['source_uid']!=source['source_uid']:continue
            for t in range(event['core_frame_start']+12,event['core_frame_end']-12):
                if not valid[t-12:t+13].all():continue
                clip=motion[t-12:t+13].copy()
                rot=matrices(clip[:,79:139].reshape(25,10,6)).astype(float)
                candidates.append(dict(source_uid=source['source_uid'],event=event['event_uid'],raw_center=t,source_sha256=source['source_sha256'],
                    clip=clip,entry_rotation=rot[0],exit_rotation=rot[-1],activity_rad_s=float(angle_difference(rot[:-1],rot[1:]).mean()*30)))
    eligible=[{k:c[k] for k in ('source_uid','event','raw_center','activity_rad_s')} for c in candidates]
    write(out/'eligible.json',eligible)
    chosen=[];anchors=[];exit_rotation=None
    for anchor in q['anchors']:
        tick=anchor['frame']
        slot=next(s for s in slots['slots'] if s['music_start']<=tick<s['music_end'])
        candidate,reason=select_supported_phrase(candidates,slot['target_motion_density'],exit_rotation)
        exit_rotation=candidate['exit_rotation']
        clip=candidate['clip']
        file=out/('clip_%d.npy'%tick)
        np.save(file,clip)
        chosen.append(dict(tick=tick,music_slot=slot['slot'],music_event=slot['music_event'],score_components=reason,
            event=candidate['event'],source_uid=candidate['source_uid'],raw_source_sha256=candidate['source_sha256'],raw_center=candidate['raw_center']))
        anchors.append(dict(frame=tick,source=dict(path=str(file),sha256=sha(file)),source_frame=12,
                            catalog_id=candidate['event'],pose=clip[12].tolist(),joints=list(range(12,22))))
    write(out/'selection.json',dict(catalog_sha256=sha(catalog_path),slots_sha256=sha(slots_path),eligible=len(eligible),chosen=chosen))
    run(dict(q,anchors=anchors,output=str(out/'generation')))
if __name__=='__main__':main()
