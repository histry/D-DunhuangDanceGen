"""Freeze all three models before exposed DEV and fixed music interventions."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.context_r21.sampler import sample
from .engine import load,cache_data,state,model_meta,probes,music_state,control_cases

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();bindings=[];q_scores={}
    population=read(root/"data/result.json")
    for arm in cfg["arms"]:
        model=read(root/f"models/{arm}/result.json")
        w,d,data,p,h=load(cfg,root,arm,model["checkpoint"]);cache=cache_data(cfg,root,d,data);before=model_meta(w,h)
        scores,arrays=probes(cfg,cache,p,controls=True,dev=True);qp=out/f"q_{arm}.npz";np.savez_compressed(qp,**arrays)
        q_scores[arm]=dict(scores=scores,predictions=ref(qp))
        if replay:
            old=read(Path(replay)/"result.json")["q_scores"][arm];assert scores==old["scores"];same_arrays(qp,checked(old["predictions"]))
        for case in population["train_probe"]+population["dev_cases"]:
            for variant in cfg["eval_variants"]:
                modes=("correct","music_swapped","music_zero") if case in control_cases(cache) and variant==0 else ("correct",)
                for mode in modes:
                    s,original,_=state(cfg,cache,case,519,variant=variant);s,music_binding=music_state(cache,s,case,mode)
                    seed=cfg["seed"]+70000+case*100+variant
                    arrays,receipt=sample(cfg,cache,p,s,seed,detailed=case==population["train_probe"][0] and variant==0 and mode=="correct",score_mask=original)
                    name=f"{arm}_{case}_{variant}_{mode}";folder=out/name;folder.mkdir()
                    trace=folder/"trace.npz";np.savez_compressed(trace,**arrays);write(folder/"sampler.json",receipt)
                    if replay:
                        same_arrays(trace,Path(replay)/name/"trace.npz");assert receipt==read(Path(replay)/name/"sampler.json")
                    row=next(r for r in cache["rows"] if r["case"]==case)
                    bindings.append(dict(arm=arm,case=case,variant=variant,control=mode,seed=seed,role=row["role"],layout="joint",
                        source_group=row["left"]["source_group"],trace=ref(trace),sampler=ref(folder/"sampler.json"),data=row["arrays"],music_control_binding=music_binding))
                    print("R22_ROLLOUT",name,flush=True)
        assert model_meta(w,h)==before;p.close()
    write(out/"result.json",dict(bindings=bindings,q_scores=q_scores,sampler_calls=len(bindings),optimizer_steps=0,whole_song_generations=0,operational_qualified=False))
    write(out/"protocol.json",dict(config=cfg,code_commit=commit(),final_models={a:ref(root/f"models/{a}/result.json") for a in cfg["arms"]},
        dev_used_for_selection=False,controls="Music swapped within role or zero features; not artistic wrong-dance labels"))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)
