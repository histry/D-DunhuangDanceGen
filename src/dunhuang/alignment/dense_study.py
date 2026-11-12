"""Separate dense coverage challenge; unchanged R1 gate, conditional fixed-path scoring."""
import argparse,copy,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import write,sha,checked
from .clock_data import read,ref
from .dense_coverage import build
from .local_identifiability import frozen_head,source_inputs
from .event_study import make_head
from .sampling import group_schedule,coverage
from .event_metrics import multi_positive_loss
from .content_discrimination import pair_result,template_paths,fixed_score,metrics

VIEWS={"tcn30":"music_tcn30","tcn15":"music_tcn15","transformer15":"music_transformer15","tcn30_random":"music_tcn30"}

def evaluate(head,bank,rows,conditional,templates,role):
    ids=np.array([i for i,r in enumerate(rows) if r["role"]==role and conditional["eligible"][i]],int)
    scores=[]
    with torch.no_grad():
        for start in range(0,len(ids),32):
            q=ids[start:start+32];c=conditional["candidates"][q];width=c.shape[1]
            r=pair_result(head,bank,np.repeat(q,width),c.reshape(-1))
            scores.append(fixed_score(r,templates[q].reshape(-1,*templates.shape[-2:]),head.band_seconds)["mean_content"].cpu().numpy().reshape(len(q),width))
    values=np.concatenate(scores) if scores else np.empty((0,4))
    return metrics(values,conditional["positive"][ids],conditional["valid"][ids],[rows[i] for i in ids]),values

def run(source,clock,out,config,replay_from=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    if subprocess.run(["git","merge-base","--is-ancestor",config["frozen_baseline"],"HEAD"],capture_output=True).returncode:raise ValueError("Frozen baseline ancestor differs")
    if config["strict"]!=read("configs/local_identifiability.json")["strict"]:raise ValueError("Historical nuisance thresholds changed")
    if config["test_evaluations"] or config["encoder_training"] or config["planner_routing"] or config["new_blind_source"]:raise ValueError("Shadow boundaries differ")
    source,clock,_,_,_,_,_,refs=source_inputs(source,clock)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows,bank,conditional,evidence=build(source,clock,out/"data",config)
    gate=all(v["population_sufficient"] for v in evidence["populations"].values())
    reference,payload=frozen_head(source/"path_synthetic/head.pt");null=reference.null_logit.detach().clone()
    templates=np.zeros((len(rows),4,12,20),bool);ids=np.flatnonzero(conditional["eligible"])
    for start in range(0,len(ids),64):
        q=ids[start:start+64];c=conditional["candidates"][q]
        templates[q]=template_paths(bank,np.repeat(q,4),c.reshape(-1),reference.band_seconds,float(.05+.9*null.sigmoid())).reshape(len(q),4,12,20)
    np.save(out/"clock_templates.npy",templates)
    protocol=dict(schema="registered_dense_challenge_v1",code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        config=config,input_references=refs,gate_passed=gate,source_groups_inherited=True,windows_independent=False,
        fixed_null=float(null),fixed_band_seconds=reference.band_seconds,selection=config["selection"],operational_qualified=False)
    write(out/"protocol.json",protocol)
    results={};updates=0
    if gate:
        selected=[dict(r,role=r["role"] if conditional["eligible"][i] else "excluded") for i,r in enumerate(rows)]
        schedule=group_schedule(selected,config["steps"],config["batch"],config["seed"]);np.save(out/"training_schedule.npy",schedule)
        protocol["schedule_coverage"]=coverage(selected,schedule);write(out/"protocol.json",protocol)
        initial_states=[]
        for name in config["arms"]:
            torch.manual_seed(config["seed"]);head=make_head("strict_path",payload["config"]).cuda()
            with torch.no_grad():head.null_logit.copy_(null)
            head.null_logit.requires_grad_(False);view=dict(bank,music=bank[VIEWS[name]])
            folder=out/name;folder.mkdir()
            initial_states.append({k:v.detach().clone() for k,v in head.state_dict().items()})
            if replay_from:
                saved=torch.load(Path(replay_from)/name/"head.pt",map_location="cuda",weights_only=True)
                if saved["config"]!=config:raise ValueError("Dense checkpoint protocol differs")
                head.load_state_dict(saved["state"])
            else:
                optimizer=torch.optim.AdamW([head.music.weight,head.motion.weight],lr=config["learning_rate"]);history=[]
                for step,q in enumerate(schedule,1):
                    c=conditional["candidates"][q];width=c.shape[1]
                    r=pair_result(head,view,np.repeat(q,width),c.reshape(-1))
                    score=fixed_score(r,templates[q].reshape(-1,12,20),head.band_seconds)["mean_content"].reshape(len(q),width)
                    p=conditional["positive"][q].copy();v=conditional["valid"][q]
                    if name=="tcn30_random":
                        rng=np.random.default_rng(config["seed"]+step)
                        for i in range(len(q)):
                            slots=np.flatnonzero(v[i]);p[i,slots]=np.roll(p[i,slots],int(rng.integers(1,len(slots))))
                    loss=multi_positive_loss(score,torch.as_tensor(p,device="cuda"),torch.as_tensor(v,device="cuda"),config["score_temperature"])
                    if not torch.isfinite(loss):raise ValueError("Dense conditional loss nonfinite")
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([head.music.weight,head.motion.weight],1.);optimizer.step();updates+=1
                    if step%50==0:
                        history.append(dict(step=step,loss=float(loss.detach())));write(folder/"history.json",history);print("DENSE_TRAIN",name,step,flush=True)
                torch.save(dict(state=head.state_dict(),config=config,step=config["steps"],operational_qualified=False),folder/"head.pt")
            if not torch.equal(null,head.null_logit.detach()):raise ValueError("NULL changed")
            arms={}
            for role in ("train","val"):
                m,scores=evaluate(head,view,rows,conditional,templates,role);np.save(folder/(role+"_scores.npy"),scores);arms[role]=m
            arms["parameters"]=sum(p.numel() for p in head.parameters());arms["trainable_parameters"]=sum(p.numel() for p in head.parameters() if p.requires_grad)
            results[name]=arms
        if any(any(not torch.equal(initial_states[0][k],state[k]) for k in state) for state in initial_states[1:]):raise ValueError("Unequal initial projection states")
    result=dict(status="COMPLETED_DENSE_COVERAGE_SHADOW",windows=len(rows),recordings=len({r["recording_uid"] for r in rows}),
        strict_population=evidence["populations"],gate_passed=gate,training_status="COMPLETED_FIXED_BUDGET" if gate else "NOT_STARTED_INSUFFICIENT_CONDITIONAL_POPULATION",
        results=results,optimizer_updates=config["steps"]*len(config["arms"]) if gate else 0,
        execution_optimizer_calls=updates,thresholds_relaxed=False,original_r1_unchanged=True,new_generation_targets=False,new_blind_source=False,operational_qualified=False)
    if replay_from:
        original=Path(replay_from);old=read(original/"result.json")
        for name in ("nodes","conditional"):
            a=np.load(original/"data"/(name+".npz"));b=np.load(out/"data"/(name+".npz"))
            if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError("Dense source rebuild differs")
        a=read(original/"data/manifest.json")["rows"];b=read(out/"data/manifest.json")["rows"]
        def semantic(rows):return [dict({k:v for k,v in r.items() if k!="source_native"},source_native_sha256=r["source_native"]["sha256"]) for r in rows]
        if semantic(a)!=semantic(b):raise ValueError("Dense rebuilt source bounds differ")
        if sha(original/"clock_templates.npy")!=sha(out/"clock_templates.npy"):raise ValueError("Clock templates differ")
        if gate:
            if not np.array_equal(np.load(original/"training_schedule.npy"),np.load(out/"training_schedule.npy")):raise ValueError("Schedule differs")
            for name in config["arms"]:
                for role in ("train","val"):
                    if not np.array_equal(np.load(original/name/(role+"_scores.npy")),np.load(out/name/(role+"_scores.npy"))):raise ValueError("Saved head forward replay differs")
        comparable=dict(result,execution_optimizer_calls=old["execution_optimizer_calls"])
        if comparable!=old:raise ValueError("Dense replay metrics differ")
        write(out/"replay.json",dict(status="FRESH_FULL_SOURCE_DENSE_REBUILD_VERIFIED",optimizer_calls=0,windows_rebuilt=len(rows),head_forward_recomputed=gate,source_clock_rechecked=True))
    write(out/"result.json",result);print("DENSE_COMPLETED",len(rows),"GATE",gate,flush=True)

def main():
    p=argparse.ArgumentParser()
    for k in ("source","clock","out","config"):p.add_argument("--"+k,required=True)
    p.add_argument("--replay-from");a=p.parse_args()
    if a.replay_from:
        def forbid(*args,**kwargs):raise RuntimeError("Replay optimizer forbidden")
        torch.optim.AdamW=forbid;torch.optim.Adam=forbid;torch.optim.SGD=forbid
    run(a.source,a.clock,a.out,read(a.config),a.replay_from)
if __name__=="__main__":main()
