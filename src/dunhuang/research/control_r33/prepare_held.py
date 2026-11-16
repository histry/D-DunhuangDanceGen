"""Recompile held maps on the actual Core music clock before any held generation."""
import argparse
import numpy as np
from .common import *
from .runtime import Runtime
from .inventory import compile_holdouts

def run(cfg,root,out,replay=None):
    out=fresh(out);forbid_optimizers()
    rt=Runtime(cfg);_,_,_,z=population_data()
    arrays,records,assets=compile_holdouts(cfg,rt,z["music"])
    old=read(Path(root)/"data/result.json")["held_records"]
    assert [r["event_uid"] for r in records]==[r["event_uid"] for r in old]
    rt.check();rt.close()
    np.savez_compressed(out/"arrays.npz",**arrays)
    result=dict(records=records,assets=assets,source_selection_changed=False,old_compilation=str(Path(root)/"data/held_observations.npz"),
        correction="Music beat channel uses Core64:160, translated to local0:96; prior0:96 map retained as superseded",
        generated_trajectories=0,optimizer_steps=0)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),arrays_sha256=sha(out/"arrays.npz")))
    if replay:
        assert result==read(Path(replay)/"result.json");same_arrays(out/"arrays.npz",Path(replay)/"arrays.npz")
    print("R33_HELD_COMPILED",len(records),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)
