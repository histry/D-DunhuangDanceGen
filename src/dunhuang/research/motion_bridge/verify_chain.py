"""Replay the raw-x0 -> hard-mask -> inverse-scaler -> graft Root chain."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.contracts import Q
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.state_bridge.common import cases

def run(generation,out,replay=None):
    out=fresh(out);items=cases(read("src/dunhuang/research/state_bridge/config.json"))
    items={x["name"]:x for x in items};g=read(Path(generation)/"result.json");records=[]
    for binding in g["bindings"]:
        item=items[binding["case"]];q=item["request"];s=np.load(checked(q["scaler"]))
        raw=np.load(checked(binding["raw_last"]));folder=Path(checked(binding["native"])).parent
        normalized=np.load(folder/"normalized.npy")[0];edited=np.load(checked(binding["native"]))
        context=np.load(checked(q["context"]));value=context["values"][0];mask=context["mask"][0]
        expected=value*s["scale"]+s["bias"]
        assert np.array_equal(normalized[60:90],raw[10:40])
        assert np.max(abs(normalized[mask]-expected[mask]))<=2e-6
        o=np.asarray(q["root_origin_native"],np.float32)
        rawroot=((raw[:,4:7]-s["bias"][4:7])/s["scale"][4:7])@Q+o
        finalroot=((normalized[:,4:7]-s["bias"][4:7])/s["scale"][4:7])@Q+o
        assert abs(edited[60:90,4:7]-finalroot[60:90]).max()<=2e-6
        assert np.array_equal(edited[:60],item["original"][:60]) and np.array_equal(edited[90:],item["original"][90:])
        for t in (60,90):
            i=t-50;k=59 if t==60 else 90
            def step(x,i):return float(np.linalg.norm(x[i]-x[i-1]))
            records.append(dict(case=binding["case"],arm=binding["arm"],seam=t,
                raw_internal_root_step_m=step(rawroot,i),
                raw_known_root_error_m=float(np.linalg.norm(rawroot[k-50]-item["original"][k,4:7])),
                after_hard_root_step_m=step(finalroot,t),after_native_graft_root_step_m=step(edited[:,4:7],t),
                hard_to_graft_step_difference_m=abs(step(finalroot,t)-step(edited[:,4:7],t))))
    assert all(r["hard_to_graft_step_difference_m"]<=2e-6 for r in records)
    result=dict(records=records,raw_unknown_equals_final_exact=True,known_overwrite_within_codec_tolerance=True,
        protected_native_exact=True,post_codec_additional_root_jump=False,optimizer_steps=0,new_neural_generations=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),generation=ref(Path(generation)/"result.json"),atol_m=2e-6))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R12_ROOT_CHAIN_VERIFIED",len(records),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--generation",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(a.generation,a.out,a.replay_from)
