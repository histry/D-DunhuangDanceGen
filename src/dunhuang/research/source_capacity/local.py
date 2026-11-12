"""Frozen tokens; no encoder updates, no candidate or population changes."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.generation.targets import digest
from dunhuang.research.source_tasks.content_model import ContentModel
from dunhuang.research.source_tasks.content import encode_val,score_val,metrics
from .common import read,commit,fresh,forbid_optimizers,same_arrays,frozen,weight
from .paths import operators

def run(config,out,replay_from=None):
    out=fresh(out);root,pub,_,_=frozen(config);old=read(root/"content/protocol.json")
    data=read(checked(old["prepared"]));manifest=read(checked(data["manifest"]));rows=manifest["rows"]
    checked(data["normalizer"]);arrays=dict(np.load(checked(data["observations"])))
    cs={name:dict(np.load(checked(value))) for name,value in manifest["candidate_files"].items()}
    ids=np.flatnonzero([r["role"]=="val" for r in rows]);pairs=sorted({(int(i),int(j)) for c in cs.values() for i in ids if c["eligible"][i] for j in c["candidates"][i,c["valid"][i]]})
    permutation=np.random.default_rng(config["seed"]).permutation(120)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False);forbid_optimizers()
    results={};scores_all={};clock_score=None;models={}
    for arm in config["local"]["models"]:
        rate="dual" if arm=="dual_shuffle" else arm;model=ContentModel(rate).cuda()
        file=weight(root,pub,"content",arm);saved=torch.load(file,map_location="cuda",weights_only=True)
        if saved["task_binding"]!=data["task_binding"] or saved["normalizer_sha256"]!=data["normalizer"]["sha256"]:raise ValueError("Frozen encoder/normalizer binding differs")
        model.load_state_dict(saved["state"]);model.eval();model.requires_grad_(False);before=digest(model);models[arm]=ref(file)
        banks,lookup=encode_val(model,arrays,rows)
        token_file=out/(arm+"_tokens.npz");np.savez_compressed(token_file,**{branch+"_"+str(i):t.cpu().numpy() for branch,values in banks.items() for i,t in enumerate(values)},lookup=lookup)
        pair_scores={}
        with torch.no_grad():
            for start in range(0,len(pairs),config["local"]["batch"]):
                block=np.array(pairs[start:start+config["local"]["batch"]]);a=torch.as_tensor(lookup[block[:,0]],device="cuda");b=torch.as_tensor(lookup[block[:,1]],device="cuda")
                m=tuple(v[a] for v in banks["music"]);d=tuple(v[b] for v in banks["motion"])
                op=operators(m[1],d[1],m[2],d[2],config["local"],clock_score)
                if clock_score is None:clock_score=op["S3"][:1].detach()
                swapped=operators(m[1],d[1][:,permutation],m[2],d[2],config["local"],clock_score)["S2"]
                global_score=(m[0]*d[0]).sum(-1)
                values={**op,"global":global_score,"scrambled":swapped}
                numpy_values={k:v.cpu().numpy() for k,v in values.items()}
                for n,key in enumerate(map(tuple,block.tolist())):pair_scores[key]={k:float(v[n]) for k,v in numpy_values.items()}
                if start%2048==0:print("LOCAL_PATH",arm,start,len(pairs),flush=True)
        tasks={}
        for task,c in cs.items():
            baseline=score_val(model,banks,lookup,c,rows,old["config"]["content"])
            old_scores=dict(np.load(root/"content"/arm/"scores.npz"))[task]
            if not np.array_equal(baseline,old_scores):raise ValueError("Original S0 frozen-score replay differs")
            all_scores={name:np.zeros(c["valid"].shape,np.float32) for name in ("S0","S1","S2","S3","global","local0","local2","scrambled")}
            for i in ids:
                if not c["eligible"][i]:continue
                for slot in np.flatnonzero(c["valid"][i]):
                    v=pair_scores[(int(i),int(c["candidates"][i,slot]))]
                    for name in ("S0","S1","S2"):all_scores[name][i,slot]=v["global"]+config["local"]["local_weight"]*v[name]
                    all_scores["S3"][i,slot]=v["S3"];all_scores["global"][i,slot]=v["global"]
                    all_scores["local0"][i,slot]=v["S0"];all_scores["local2"][i,slot]=v["S2"]
                    all_scores["scrambled"][i,slot]=v["global"]+config["local"]["local_weight"]*v["scrambled"]
            valid_scores=c["valid"]&c["eligible"][:,None]&np.array([r["role"]=="val" for r in rows])[:,None]
            if not np.allclose(all_scores["S0"][valid_scores],baseline[valid_scores],rtol=0,atol=2e-6):raise ValueError("S0 common-grid score reconstruction differs")
            all_scores["S0"]=baseline
            comparisons={"S1_minus_S0":("S1","S0"),"S2_minus_S0":("S2","S0"),"S2_minus_global":("S2","global"),
                "S2_minus_scrambled":("S2","scrambled"),"local2_minus_time":("local2","S3"),"local0_minus_time":("local0","S3")}
            tasks[task]={name:metrics(all_scores[a],all_scores[b],c,rows,config["local"]) for name,(a,b) in comparisons.items()}
            tasks[task]["S2_minus_nuisance"]=metrics(all_scores["S2"],c["train_nuisance_linear"],c,rows,config["local"])
            tasks[task]["uniform_random_group_R1"]=float(np.mean([np.mean([1/int(c["valid"][i].sum()) for i in ids if c["eligible"][i] and rows[i]["source_group"]==group]) for group in sorted({rows[i]["source_group"] for i in ids if c["eligible"][i]})]))
            tasks[task]["time_only_tie_policy"]="pessimistic; S3 is not random chance"
            for name,value in all_scores.items():scores_all[arm+"__"+task+"__"+name]=value
        results[arm]=tasks
        if digest(model)!=before:raise ValueError("Frozen content parameters changed")
        del model,banks;torch.cuda.empty_cache()
    for task,c in cs.items():
        results["dual"][task]["S2_minus_frozen_shuffle"]=metrics(scores_all["dual__"+task+"__S2"],scores_all["dual_shuffle__"+task+"__S2"],c,rows,config["local"])
    result=dict(models=results,optimizer_calls=0,statistics_fit_calls=0,unique_val_pairs=len(pairs),test_tensor_reads=0,operational_qualified=False,new_blind=False)
    np.savez_compressed(out/"scores.npz",**scores_all);write(out/"result.json",result)
    write(out/"protocol.json",dict(code_commit=commit(),config=config,weights=models,prepared=old["prepared"],normalizer=data["normalizer"],
        candidate_files=manifest["candidate_files"],original_score_exact=True,common_grid_fps=30,frames=120,optimizer_calls=0))
    if replay_from:
        oldroot=Path(replay_from)
        if read(oldroot/"result.json")!=result:raise ValueError("Local-path readouts differ")
        same_arrays(oldroot/"scores.npz",out/"scores.npz")
        for arm in config["local"]["models"]:same_arrays(oldroot/(arm+"_tokens.npz"),out/(arm+"_tokens.npz"))
        write(out/"replay.json",dict(status="FRESH_FROZEN_TOKEN_PATH_REPLAY",optimizer_calls=0,models_verified=4))
    print("LOCAL_PATH_COMPLETED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)
