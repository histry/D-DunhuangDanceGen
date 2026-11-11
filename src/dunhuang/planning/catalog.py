import json
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write,sha,pinned
from dunhuang.contracts import MotionAdapter,matrices
from dunhuang.alignment.semantics import annotate
from scipy.spatial.transform import Rotation

def build(assets,destination):
    import sys
    root=pinned(assets['legacy_root'],assets['legacy_commit']);sys.path.insert(0,str(root))
    from motion.retrieval_generation import decode_continuous_source
    from evaluation.paired_alignment_observations_v3 import motion_from_edge151
    catalog_file=checked(assets['catalog_source'])
    catalog=json.loads(catalog_file.read_text())
    scaler=np.load(checked(assets['scaler']))
    destination.mkdir(parents=True,exist_ok=False)
    sources={}
    for row in catalog['source_records']:
        file=Path(assets['motion_root'])/Path(row['raw_receipt']['path']).name
        checked(dict(path=str(file),sha256=row['source_sha256']))
        n=row['raw_frames']
        motion,_=decode_continuous_source(file,row['discovery_metadata'],(0,n),known_rows=np.ones(n,bool))
        path=destination/(row['source_uid']+'.npy');np.save(path,motion)
        sources[row['source_uid']]=dict(motion=motion,ref=dict(path=str(path.resolve()),sha256=sha(path)),metadata=row)
    entries=[]
    for row in catalog['events']:
        source=sources[row['source_uid']]
        a,b=row['core_frame_start'],row['core_frame_end']
        x=source['motion'][a:b]
        rotation=matrices(x[:,7:].reshape(-1,24,6)).astype(float)
        velocity=Rotation.from_matrix((rotation[:-1].swapaxes(-1,-2)@rotation[1:]).reshape(-1,3,3)).magnitude().reshape(-1,24)*30
        energy=velocity[:,12:22].mean(1)
        if len(x)<25:continue
        motion_peak=int(energy.argmax()+1)
        peak=int(np.clip(motion_peak,max(13,round(.35*len(x))),min(len(x)-14,round(.65*len(x)))))
        reference=MotionAdapter.to_edge(x[peak-12:peak+13].copy())
        support=bool(np.all(abs(reference[:,79:127]*scaler['scale'][79:127]+scaler['bias'][79:127])<=1+32*np.finfo(np.float32).eps))
        values,mask,_=motion_from_edge151(x)
        feature_file=destination/(row['event_uid']+'.npz');np.savez_compressed(feature_file,motion=values,mask=mask)
        entries.append(dict(motion_features=dict(path=str(feature_file.resolve()),sha256=sha(feature_file)),event_uid=row['event_uid'],source_uid=row['source_uid'],
            recording_uid=row['recording_uid'],source_sha256=source['metadata']['source_sha256'],
            source_interval=[a,b],duration_frames=b-a,reference=source['ref'],center=a+peak,
            activity=float(energy.mean()),entry=rotation[0,12:20].tolist(),exit=rotation[-1,12:20].tolist(),
            phases=dict(motion_peak=motion_peak,primitive_center=peak,slowest=int(energy.argmin()+1)),
            semantic=annotate(row['source_uid']),feasibility=dict(source='PASS',temporal8='PASS' if support else 'FAIL',soft_reference='PASS',native='UNKNOWN')))
    for row in entries:row['activity_percentile']=float(np.mean([e['activity']<row['activity'] for e in entries]))
    write(destination/'catalog.json',dict(schema='event_catalog_v1',fps=30,source_catalog_sha256=sha(catalog_file),
        scaler_sha256=assets['scaler']['sha256'],entries=entries))
    return destination/'catalog.json'
