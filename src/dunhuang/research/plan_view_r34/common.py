from pathlib import Path
import hashlib,json
import numpy as np
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,fresh,commit,forbid_optimizers,same_arrays

def identity(audio_sha,event,window_start=None):
    payload=dict(audio_sha256=audio_sha,event_uid=event["event_uid"],source_sha256=event["source_sha256"],
        source_interval=event["source_interval"],music_interval=event["music_interval"],time_warp=event["time_warp"],window_start=window_start)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def resolve_audio(record):
    candidates=[Path(record["path"]),Path("inputs/r7_audio_recovery")/Path(record["path"]).name]
    for p in candidates:
        if p.exists() and sha(p)==record["sha256"]:return p
    raise ValueError("Registered full audio bytes unavailable; never substitute by filename")

def relative_fk(R,J,parents):
    R=np.asarray(R,np.float64);J=np.asarray(J,np.float64)
    p=[np.zeros((len(R),3))];world=[R[:,0]]
    for j in range(1,22):
        k=int(parents[j]);p.append(p[k]+(world[k]@(J[j]-J[k])[None,:,None])[...,0]);world.append(world[k]@R[:,j])
    return np.stack(p,1).astype(np.float32)
