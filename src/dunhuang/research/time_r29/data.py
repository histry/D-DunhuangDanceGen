"""Eight FineDance windows grouped by recording; use authenticated R28 preprocessing outputs."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import runtime
from dunhuang.research.state_r27.common import read,write,sha,commit,fresh,forbid_optimizers,same_arrays

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();parent=Path(cfg["r28"])
    assert sha(parent/"artifacts_manifest.json")==cfg["r28_manifest_sha256"]
    manifest=read(parent/"artifacts_manifest.json")["files"]
    d,n,s,f,e,info,a=runtime(cfg);a.close();records=[]
    for role,ids in (("train",cfg["fine_train_recordings"]),("holdout",cfg["fine_holdout_recordings"])):
        for name in ids:
            paths=[parent/"data/finedance/motion319"/(name+".npy"),parent/"fine_audio"/(name+".npy")]
            for p in paths:assert sha(p)==manifest[str(p.relative_to(parent))]["sha256"]
            motion=np.load(paths[0]);music=np.load(paths[1]);length=min(len(motion),len(music))
            for start in cfg["fine_offsets"]:
                assert start+256<=length
                raw=motion[start:start+256,:139].astype(np.float32).copy();raw[:,[4,6]]-=raw[:1,[4,6]]
                with torch.no_grad():
                    tensor=torch.tensor(raw[None],device="cuda");target=n.normalize(tensor.clone());points=f(tensor,s)[:,:,:22]
                path=out/f"{name}_{start}.npz";np.savez_compressed(path,model_raw=raw,model_target=target[0].cpu().numpy(),model_points=points[0].cpu().numpy(),music=music[start:start+256].astype(np.float32))
                records.append(dict(item=f"{name}:{start}",recording=name,role=role,start_frame=start,end_frame=start+256,arrays=path.name,sha256=sha(path),
                    source_motion_sha256=sha(paths[0]),source_music_sha256=sha(paths[1]),fps=30,clock="OFFICIAL_SEQUENCE_COMMON_TRIM_PREFIX",pretraining_exposure_unknown=True))
    assert not set(cfg["fine_train_recordings"])&set(cfg["fine_holdout_recordings"])
    result=dict(records=records,training_recordings=2,holdout_recordings=2,windows_per_recording=2,adjacent_windows_not_independent=True,
        optimizer_steps=0,full_native_status="NOT_EVALUABLE",operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),checkpoint=info))
    if replay:
        assert result==read(Path(replay)/"result.json")
        for p in out.glob("*.npz"):same_arrays(p,Path(replay)/p.name)
    print("R29_FINE_WINDOWS",len(records),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)
