"""Independent read-only mapping parity; no candidate replacement or output editing."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,write,sha
from dunhuang.contracts import matrices
from dunhuang.generation.soft_reference import resample_native
from dunhuang.planning.time_map import source_positions
from dunhuang.research.source_capacity.common import read,commit,forbid_optimizers

def run(root,out,replay=None):
    root=Path(root);out=Path(out);assert not out.exists();forbid_optimizers()
    pre=read(root/"main/protocol.json");cfg=pre["config"];z=np.load(checked(pre["arrays"]))
    pairs=[read(checked(pre["plans"][key])) for key in ("A","B")];records=[]
    for i,plan in enumerate(pairs):
        e=plan["events"][0];T=plan["frames"];a=cfg["core_start"];positions=source_positions(e,np.arange(T))
        source=resample_native(np.load(checked(e["reference"])),positions)
        R=matrices(source[:,7:].reshape(T,24,6))[:,:22]
        mapped=matrices(z["source139"][i,a:a+T,7:].reshape(T,22,6),row=True)
        error=Rotation.from_matrix((R.swapaxes(-1,-2)@mapped).reshape(-1,3,3)).magnitude()
        assert error.max()<1e-6
        frames=np.array(sorted({p["frame"] for p in e["full_phases"]}))
        warp=np.array(e["time_warp"]);ticks=np.interp(frames,warp[:,0],warp[:,1])+a
        assert (np.diff(ticks)>0).all()
        records.append(dict(event_uid=e["event_uid"],source_interval=e["source_interval"],target_core=[a,a+T],
            mapped_pose_max_rad=float(error.max()),mapped_pose_mean_rad=float(error.mean()),unique_full_phase_frames=len(frames),
            phase_order_preserved=True,time_map_rate_bounds=[float(np.min(np.diff(warp[:,1])/np.diff(warp[:,0]))),float(np.max(np.diff(warp[:,1])/np.diff(warp[:,0])))],
            Root_translation_and_C4_prompted=False,terminal22_23_rotations_observed=False))
    result=dict(records=records,main_protocol_sha256=sha(root/"main/protocol.json"),input_arrays_sha256=pre["arrays"]["sha256"],
        read_only=True,optimizer_steps=0,model_outputs_changed=False)
    if replay:assert result==read(replay)
    write(out,result);print("R24_MAPPING_AND_PHASE_ORDER_VERIFIED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(a.root,a.out,a.replay_from)
