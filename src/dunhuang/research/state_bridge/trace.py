"""Fresh frozen-C2 A0 trajectories with same-state A0..A5 probes and exact replay."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from dunhuang.io import write,checked
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.primitive_reperformance.environment import worker_environment
from dunhuang.research.primitive_reperformance.study import ARRAYS
from .common import cases

def run(cfg,audit,out,replay=None):
    out=fresh(out);assert read(Path(audit)/"result.json")["oracle_passed"]
    bindings=[]
    for item in cases(cfg):
        folder=out/item["name"];folder.mkdir();q=dict(item["request"])
        q.update(output=str(folder.resolve()),research_r11=True,r11_wrong_context=item["wrong_context"],r11_arms=cfg["arms"])
        write(folder/"request.json",q)
        with (folder/"worker.log").open("w") as log:
            subprocess.run([q["edge_python"],"-m","dunhuang.research.state_bridge.worker",str(folder/"request.json")],check=True,stdout=log,stderr=subprocess.STDOUT,env=worker_environment())
        with (folder/"physical.log").open("w") as log:
            subprocess.run([q["native_python"],"-m","dunhuang.physical",str(folder/"request.json")],check=True,stdout=log,stderr=subprocess.STDOUT,env=worker_environment())
        g=read(folder/"generation.json");old=read(item["parent_folder"]/"generation.json")
        assert g["weights_before"]==g["weights_after"] and g["sampler"]["noise_sha256"]==old["sampler"]["noise_sha256"]
        assert g["sampler"]["final_normalized_anchor_max_abs"]==0
        for f in ARRAYS:
            assert np.array_equal(np.load(folder/f),np.load(item["parent_folder"]/f)),(item["name"],f)
        data=np.load(folder/"trace.npz")
        assert data["prediction"].shape==(50,6,50,151) and data["latent"].shape==(50,50,151)
        if replay:
            other=Path(replay)/item["name"]
            for key in data.files:assert np.array_equal(data[key],np.load(other/"trace.npz")[key])
            assert read(folder/"trace.json")==read(other/"trace.json") and read(folder/"physical.json")==read(other/"physical.json")
            for f in ARRAYS:assert np.array_equal(np.load(folder/f),np.load(other/f))
        bindings.append(dict(case=item["name"],trace=ref(folder/"trace.npz"),trace_info=ref(folder/"trace.json"),request=ref(folder/"request.json"),
            reference=ref(item["parent_folder"]/"generation.json"),wrong_case=item["wrong_case"]))
        print("R11_TRACE_A0_EQUIVALENT",item["name"],flush=True)
    result=dict(cases=len(bindings),samplings=len(bindings),counterfactual_forward_predictions=len(bindings)*50*6,optimizer_steps=0,full_counterfactual_rollouts=0,operational_qualified=False,new_whole_song_generations=0)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,bindings=bindings))
    if replay:assert result==read(Path(replay)/"result.json")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--audit",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.audit,a.out,a.replay_from)
