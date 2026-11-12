"""Fresh-process full observation/model replay. Optimizers are forbidden."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import write,sha,checked
from .clock_data import read
from .event_data import prepare
from .event_study import make_head,evaluate,path_diagnostics,baselines,probes,operator_audit,load_dense,donor_indices,synthetic_samples,shift_evaluation
from .event_perturb import metrics as synthetic_metrics
from .clock_experiment import digest

def compare_archive(old,new):
    a=np.load(old);b=np.load(new)
    if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError("Scientific array replay differs: "+str(old))
    return len(a.files)

def run(source,out,replay_out):
    source=Path(source);out=Path(out);replay_out=Path(replay_out).resolve();replay_out.mkdir(parents=True,exist_ok=False)
    def forbidden(*args,**kwargs):raise RuntimeError("Optimizer forbidden in fresh replay")
    torch.optim.AdamW=forbidden;torch.optim.Adam=forbidden;torch.optim.SGD=forbidden
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    protocol=read(out/"protocol.json");config=protocol["config"]
    for name,h in read(out/"preflight.json")["files"].items():
        if sha(out/name)!=h:raise ValueError("Preflight seal differs")
    checked(protocol["source"]);assets=read(checked(protocol["assets"]))
    manifest=read(out/"data/manifest.json")
    for k in ("source_manifest","source_encodings","source_observations","source_training_schedule","global_checkpoint"):checked(manifest[k])
    for item in read(out/"data/source_seals.json"):checked(item)
    rebuilt=prepare(source,replay_out/"data",config,assets)
    array_fields=0
    for name in ("events","probes","operator"):array_fields+=compare_archive(out/"data"/(name+".npz"),replay_out/"data"/(name+".npz"))
    for name in ("selected_events.json","weak_evidence.json","source_seals.json"):
        if sha(out/"data"/name)!=sha(replay_out/"data"/name):raise ValueError("Observation sidecar replay differs")
    data=dict(np.load(out/"data/events.npz"));bank={k:torch.as_tensor(v,device="cuda") for k,v in data.items()}
    rows=rebuilt["rows"];bank["donors"]=torch.as_tensor(donor_indices(rows,data["mv"].any(-1)&data["dv"].any(-1)),device="cuda");ids=[i for i,r in enumerate(rows) if r["role"]=="val" and data["eligible"][i]]
    dense=load_dense(source);samples=synthetic_samples(dense,ids,donor_indices(rows),config)
    original_samples=np.load(out/"synthetic_truth.npz")
    for key in original_samples.files:
        if not np.array_equal(original_samples[key],np.stack([s[key] for s in samples])):raise ValueError("Synthetic identity/warp recipe differs")
    validation={}
    for mode in config["arms"]:
        payload=torch.load(out/mode/"head.pt",map_location="cuda",weights_only=True)
        if payload["protocol_sha256"]!=sha(out/"protocol.json") or payload["data_sha256"]!=sha(out/"data/events.npz"):raise ValueError("Head binding differs")
        head=make_head(mode,config).cuda();head.load_state_dict(payload["state"]);before=digest(head)
        metrics,measure=evaluate(head,mode,bank,ids,rows)
        original=np.load(out/mode/"measurements.npz")
        if any(not np.array_equal(original[k],v) for k,v in measure.items()):raise ValueError("Fresh head scores differ: "+mode)
        saved=read(out/mode/"validation.json")
        for key in metrics:
            if metrics[key]!=saved[key]:raise ValueError("Fresh metrics differ: "+mode+"/"+key)
        if shift_evaluation(head,mode,dense,ids,config)!=saved["synthetic_shift"]:raise ValueError("Shift recovery replay differs")
        if mode!="sync_cold":
            q=synthetic_metrics(head,samples,"null_ot" if mode=="null_ot" else "strict_path")
            if q!=saved["synthetic_partial"]:raise ValueError("Synthetic NULL/warp replay differs")
        if path_diagnostics(head,mode,bank,ids)!=read(out/mode/"path_diagnostics.json"):raise ValueError("Path diagnostics differ")
        if digest(head)!=before:raise ValueError("Replay changed head")
        validation[mode]="EXACT"
    b=baselines(bank,ids,rows,source)
    old=np.load(out/"baseline_scores.npz")
    if any(not np.array_equal(old[k],v["scores"]) for k,v in b.items()):raise ValueError("Baseline replay differs")
    pr=probes(out/"data",rows,config,out,np.load(out/"probe_schedule.npy"),replay=True)
    if pr["results"]!=read(out/"probes.json")["results"]:raise ValueError("Probe metrics differ")
    op=operator_audit(source,replay_out/"data",replay_out)
    array_fields+=compare_archive(out/"operator_scores.npz",replay_out/"operator_scores.npz")
    if op!=read(out/"operator_audit.json"):raise ValueError("Operator audit differs")
    write(replay_out/"result.json",dict(status="FRESH_EVENT_IDENTIFIABILITY_REPLAY_VERIFIED",
        execution_commit=protocol["code_commit"],replay_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        protocol_sha256=sha(out/"protocol.json"),source_observations_rebuilt=True,array_fields_verified=array_fields,
        head_predictions=validation,probe_predictions="EXACT",optimizer_calls=0,test_evaluations=0,operational_qualified=False))
    write(out/"replay_verified.json",dict(path=str(replay_out/"result.json"),sha256=sha(replay_out/"result.json")))
    print("EVENT_REPLAY_VERIFIED",flush=True)

def main():
    ap=argparse.ArgumentParser()
    for k in ("source","out","replay-out"):ap.add_argument("--"+k,required=True)
    a=ap.parse_args();run(a.source,a.out,a.replay_out)
if __name__=="__main__":main()
