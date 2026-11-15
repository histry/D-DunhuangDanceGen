"""Final heldout evaluation after every model is fixed; same noise and masks."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import write,checked
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from .engine import load,cache_data,state,model_meta,probes
from .sampler import sample

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();bindings=[];q_scores={}
    population=read(root/"data/result.json");arms=[a for a in cfg["arms"] if a!="F" or population["support_admitted"]]
    for arm in arms:
        model=read(root/"models"/arm/"result.json");w,d,data,p,h=load(cfg,root,arm,model["checkpoint"]);cache=cache_data(cfg,root,d,data)
        before=model_meta(w,h);scores,arrays=probes(cfg,cache,p,controls=True,dev=True)
        path=out/("q_"+arm+".npz");np.savez_compressed(path,**arrays);q_scores[arm]=dict(scores=scores,predictions=ref(path))
        if replay:
            previous=read(Path(replay)/"result.json")["q_scores"][arm];assert scores==previous["scores"];same_arrays(path,checked(previous["predictions"]))
        cases=population["train_probe"]+population["dev_cases"]
        for case in cases:
            for variant in cfg["eval_variants"]:
                controls=("correct","left_wrong","right_wrong","both_absent") if arm=="B3" and variant==0 and case in (population["train_probe"][:2]+population["dev_cases"][:2]) else ("correct",)
                for control in controls:
                    s,original,_=state(cfg,cache,case,519,variant=variant,control=control)
                    seed=cfg["seed"]+70000+case*100+variant
                    detailed=case==population["train_probe"][0] and variant==0 and control=="correct"
                    arrays,receipt=sample(cfg,cache,p,s,seed,detailed=detailed,score_mask=original)
                    name=f"{arm}_{case}_{variant}_{control}";folder=out/name;folder.mkdir()
                    trace=folder/"trace.npz";np.savez_compressed(trace,**arrays);write(folder/"sampler.json",receipt)
                    if replay:
                        same_arrays(trace,Path(replay)/name/"trace.npz");assert receipt==read(Path(replay)/name/"sampler.json")
                    row=next(r for r in cache["rows"] if r["case"]==case)
                    bindings.append(dict(arm=arm,case=case,variant=variant,control=control,seed=seed,role=row["role"],layout=model["layout"],
                        source_group=row["left"]["source_group"],trace=ref(trace),sampler=ref(folder/"sampler.json"),data=row["arrays"]))
                    print("R21_ROLLOUT",name,flush=True)
        assert model_meta(w,h)==before;p.close()
    write(out/"result.json",dict(bindings=bindings,q_scores=q_scores,sampler_calls=len(bindings),optimizer_steps=0,whole_song_generations=0,operational_qualified=False))
    write(out/"protocol.json",dict(config=cfg,code_commit=commit(),final_models={a:ref(root/"models"/a/"result.json") for a in arms},
        dev_used_for_selection=False,evaluation_population="Train4 and exposed DEV4; two fixed observation layouts; one seed per layout"))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)
