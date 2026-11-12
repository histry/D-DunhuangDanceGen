"""R1 read-only diagnosis followed by gated, equal-capacity R2. No encoder/Native changes."""
import argparse,copy,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import write,sha,checked
from .clock_data import read,ref
from .clock_experiment import digest
from .event_study import make_head,load_dense,donor_indices,shift_evaluation
from .event_perturb import metrics as synthetic_metrics
from .event_metrics import multi_positive_loss
from .sampling import group_schedule,coverage
from .conditional_candidates import nuisance_statistics,build_candidates,validate_roles
from .content_discrimination import pair_result,template_paths,fixed_score,fit_normalization,metrics
from . import local_information_audit as information
from . import token_geometry_audit as geometry
from . import loss_gradient_audit as gradient

def setup():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise ValueError("Registered read-only encoder/gradient audit needs CUDA")

def frozen_head(path):
    payload=torch.load(path,map_location="cuda",weights_only=True)
    head=make_head("strict_path",payload["config"]).cuda();head.load_state_dict(payload["state"])
    head.eval();head.requires_grad_(False)
    return head,payload

def source_inputs(source,clock):
    source=Path(source).resolve();clock=Path(clock).resolve()
    manifest=read(source/"data/manifest.json");rows=manifest["rows"];validate_roles(rows)
    if read(source/"result.json")["status"]!="COMPLETED_EVENT_IDENTIFIABILITY_SHADOW":raise ValueError("Frozen event source incomplete")
    replay=read(checked(read(source/"replay_verified.json")))
    if replay["status"]!="FRESH_EVENT_IDENTIFIABILITY_REPLAY_VERIFIED":raise ValueError("Frozen source replay missing")
    references=[ref(source/"result.json"),ref(source/"protocol.json"),ref(source/"preflight.json"),ref(source/"replay_verified.json"),
        ref(source/"data/manifest.json"),ref(source/"data/events.npz"),ref(source/"data/probes.npz"),ref(source/"data/selected_events.json"),
        ref(source/"data/source_seals.json"),ref(source/"training_schedule.npy"),ref(source/"synthetic_truth.npz"),
        ref(source/"path_synthetic/head.pt"),ref(source/"path_weak/head.pt"),
        ref(clock/"data/manifest.json"),ref(clock/"data/observations.npz"),ref(clock/"encodings.npz")]
    for item in read(source/"data/source_seals.json"):checked(item)
    for name in ("source_manifest","source_encodings","source_observations","global_checkpoint"):checked(manifest[name])
    for item in references:checked(item)
    # Authenticate full-event sidecars before population construction and audit targets.
    for path,h in read(source/"preflight.json")["files"].items():
        if path.startswith("data/sidecars/") and sha(source/path)!=h:raise ValueError("Full-source event sidecar differs")
    bank=dict(np.load(source/"data/events.npz"));obs=dict(np.load(clock/"data/observations.npz"))
    events=read(source/"data/selected_events.json")
    return source,clock,manifest,rows,bank,obs,events,references

def preflight(source,clock,config,out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    if subprocess.run(["git","merge-base","--is-ancestor",config["frozen_baseline"],"HEAD"],capture_output=True).returncode:raise ValueError("Frozen baseline is not an ancestor")
    if config["test_evaluations"] or config["encoder_training"] or config["planner_routing"] or config["new_blind_source"]:raise ValueError("Registered Shadow boundaries differ")
    if subprocess.check_output(["git","diff","HEAD","--"],text=True):raise ValueError("Dirty tracked code")
    source,clock,manifest,rows,bank,obs,events,refs=source_inputs(source,clock)
    all_events={}
    for r in rows:
        key=r["audio"]["sha256"]
        if key not in all_events:all_events[key]=read(source/"data/sidecars"/(key+".json"))["events"]
    stats=nuisance_statistics(rows,bank,obs,all_events);conditional,evidence=build_candidates(rows,bank,stats,config["strict"])
    head,payload=frozen_head(source/"path_synthetic/head.pt");null=float(.05+.9*head.null_logit.detach().sigmoid())
    templates=np.zeros((len(rows),4,bank["mv"].shape[1],bank["dv"].shape[1]),bool)
    ids=np.flatnonzero(conditional["eligible"]);q=np.repeat(ids,4);c=conditional["candidates"][ids].reshape(-1)
    if len(ids):templates[ids]=template_paths(bank,q,c,head.band_seconds,null).reshape(len(ids),4,bank["mv"].shape[1],bank["dv"].shape[1])
    if len(ids) and np.any((templates[ids].sum((-1,-2))==0)&conditional["valid"][ids]):raise ValueError("Empty fixed template for conditional candidate")
    np.savez_compressed(out/"conditional.npz",**conditional,clock_template=templates)
    np.savez_compressed(out/"nuisance_statistics.npz",**stats)
    write(out/"candidate_evidence.json",evidence)
    gate=all(x["population_sufficient"] for x in evidence["populations"].values())
    protocol=dict(schema="local_identifiability_registered_study_v1",code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        baseline=config["frozen_baseline"],config=config,source=str(source),clock=str(clock),input_references=refs,
        source_training_commit=read(source/"protocol.json")["code_commit"],test_evaluations=0,operational_qualified=False,
        fixed_null_cost=null,fixed_band_seconds=head.band_seconds,fixed_epsilon=head.epsilon,
        strict_population=evidence["populations"],r2_population_gate_passed=gate,
        r2_skip_reason=None if gate else "INSUFFICIENT_PRE_REGISTERED_CONDITIONAL_TRAIN_OR_VAL_POPULATION",
        manual_event_labels=0,selection=config["selection"])
    write(out/"protocol.json",protocol)
    if gate:
        schedule_rows=[dict(r,role=r["role"] if conditional["eligible"][i] else "excluded") for i,r in enumerate(rows)]
        schedule=group_schedule(schedule_rows,config["steps"],config["batch"],config["seed"]);np.save(out/"training_schedule.npy",schedule)
        protocol["schedule_coverage"]=coverage(schedule_rows,schedule);write(out/"protocol.json",protocol)
    seal={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    write(out/"preflight.json",dict(status="PASSED_R1_SOURCE_AND_COHORT_PREFLIGHT",files=seal,strict_population=evidence["populations"],
        r2_enabled=gate,conditional_rules_relaxed=False,optimizer_started=False))
    print("LOCAL_PREFLIGHT",json.dumps(evidence["populations"]),"R2",gate,flush=True)

def cohorts(rows,bank,conditional,protocol):
    legacy=[i for i,r in enumerate(rows) if r["role"]=="val" and bank["eligible"][i]]
    strict=[i for i,r in enumerate(rows) if r["role"]=="val" and conditional["eligible"][i]]
    return dict(legacy=dict(ids=legacy,candidates=bank["candidates"],positive=bank["positive"],valid=bank["valid"],sufficient=True),
        strict=dict(ids=strict,candidates=conditional["candidates"],positive=conditional["positive"],valid=conditional["valid"],
            sufficient=protocol["strict_population"]["val"]["population_sufficient"]))

def global_baseline(enc,rows,cohort):
    ids=cohort["ids"]
    if not ids:return dict(status="NOT_EVALUABLE_EMPTY_POPULATION")
    candidate=cohort["candidates"][ids];values=(enc["mg"][ids,None,:]*enc["dg"][candidate]).sum(-1)
    return metrics(values,cohort["positive"][ids],cohort["valid"][ids],[rows[i] for i in ids],cohort["sufficient"])

def compare_tree(a,b):
    a=Path(a);b=Path(b);count=0
    for p in sorted(a.rglob("*")):
        if not p.is_file() or p.suffix not in (".json",".npz") or p.name=="history.json":continue
        target=b/p.relative_to(a)
        if not target.is_file():raise ValueError("Fresh scientific file missing: "+str(target))
        if p.suffix==".npz":
            x=np.load(p);y=np.load(target)
            if set(x.files)!=set(y.files) or any(not np.array_equal(x[k],y[k]) for k in x.files):raise ValueError("Scientific arrays differ: "+str(p))
            count+=len(x.files)
        elif p.suffix==".json":
            if read(p)!=read(target):raise ValueError("Scientific metrics differ: "+str(p))
    return count

def r2(source,rows,bank,conditional,protocol,cohort,out,replay_from=None):
    config=protocol["config"];out=Path(out);out.mkdir()
    if not protocol["r2_population_gate_passed"]:
        record=dict(status="NOT_STARTED_INSUFFICIENT_CONDITIONAL_POPULATION",reason=protocol["r2_skip_reason"],
            populations=protocol["strict_population"],optimizer_updates=0,rules_relaxed=False,operational_qualified=False)
        write(out/"result.json",record);return record
    reference,payload=frozen_head(source/"path_synthetic/head.pt");saved_null=reference.null_logit.detach().clone()
    schedule=np.load(Path(protocol["_original_out"])/"training_schedule.npy")
    template=conditional["clock_template"];results={};initializations={}
    for mode in config["r2_arms"]:
        folder=out/mode;folder.mkdir()
        torch.manual_seed(config["seed"])
        head=copy.deepcopy(reference) if mode!="conditional_cold" else make_head("strict_path",payload["config"]).cuda()
        with torch.no_grad():head.null_logit.copy_(saved_null)
        head.requires_grad_(True);head.null_logit.requires_grad_(False)
        initializations[mode]=dict(digest=digest(head),parameters=sum(p.numel() for p in head.parameters()),trainable_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad))
        if replay_from is not None:
            state=torch.load(Path(replay_from)/mode/"head.pt",map_location="cuda",weights_only=True)
            if state["protocol_sha256"]!=sha(Path(protocol["_original_out"])/"protocol.json"):raise ValueError("R2 checkpoint protocol differs")
            head.load_state_dict(state["state"])
        else:
            optimizer=torch.optim.AdamW([head.music.weight,head.motion.weight],lr=config["learning_rate"]);history=[]
            for step,ids in enumerate(schedule,1):
                c=conditional["candidates"][ids];n,width=c.shape;q=np.repeat(ids,width)
                r=pair_result(head,bank,q,c.reshape(-1))
                scores=r["score"] if mode=="path_retrieval_warm" else fixed_score(r,template[ids].reshape(-1,*template.shape[-2:]),head.band_seconds)["mean_content"]
                p=conditional["positive"][ids].copy();v=conditional["valid"][ids]
                if mode=="conditional_random":
                    rng=np.random.default_rng(config["seed"]+step)
                    for i in range(len(ids)):
                        slots=np.flatnonzero(v[i]);p[i,slots]=np.roll(p[i,slots],int(rng.integers(1,len(slots))))
                loss=multi_positive_loss(scores.reshape(n,width),torch.as_tensor(p,device="cuda"),torch.as_tensor(v,device="cuda"),config["score_temperature"])
                if not torch.isfinite(loss):raise ValueError("Nonfinite conditional content objective")
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([head.music.weight,head.motion.weight],1.);optimizer.step()
                if step%50==0:
                    history.append(dict(step=step,loss=float(loss.detach())));write(folder/"history.json",history);print("LOCAL_R2",mode,step,flush=True)
            torch.save(dict(state=head.state_dict(),protocol_sha256=sha(Path(protocol["_original_out"])/"protocol.json"),step=config["steps"],qualified=False,operational_qualified=False),folder/"head.pt")
        if not torch.equal(head.null_logit.detach(),saved_null):raise ValueError("R2 altered fixed NULL")
        mode_results={}
        for label,co in cohort.items():mode_results[label]=geometry.evaluate(head,bank,rows,co,folder/label,config)
        old_config=read(source/"protocol.json")["config"];samples_np=np.load(source/"synthetic_truth.npz")
        samples=[{k:samples_np[k][i] for k in samples_np.files} for i in range(len(samples_np["target"]))]
        safety=dict(synthetic=synthetic_metrics(head,samples,"strict_path"),
            shift=shift_evaluation(head,"strict_path",load_dense(protocol["clock"]),cohort["legacy"]["ids"],old_config))
        write(folder/"safety.json",safety);mode_results["safety"]=safety;results[mode]=mode_results
    record=dict(status="COMPLETED_EQUAL_CAPACITY_R2_SHADOW",results=results,initializations=initializations,
        optimizer_updates=0 if replay_from is not None else config["steps"]*len(config["r2_arms"]),
        null_frozen=True,test_evaluations=0,new_blind_source=False,operational_qualified=False)
    write(out/"result.json",record);return record

def run(out,replay_out=None):
    out=Path(out).resolve();protocol=read(out/"protocol.json");config=protocol["config"]
    for p,h in read(out/"preflight.json")["files"].items():
        if sha(out/p)!=h:raise ValueError("Registered local audit seal differs")
    for item in protocol["input_references"]:checked(item)
    if replay_out is None and subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()!=protocol["code_commit"]:raise ValueError("Execution commit differs")
    if replay_out is not None:
        def forbid(*a,**k):raise RuntimeError("Replay optimizer forbidden")
        torch.optim.AdamW=forbid;torch.optim.Adam=forbid;torch.optim.SGD=forbid
    setup();source,clock,manifest,rows,bank,obs,events,refs=source_inputs(protocol["source"],protocol["clock"])
    dest=out/"scientific" if replay_out is None else Path(replay_out).resolve()/"scientific"
    dest.mkdir(parents=True,exist_ok=False);cohort=cohorts(rows,bank,dict(np.load(out/"conditional.npz")),protocol)
    head,payload=frozen_head(source/"path_synthetic/head.pt");initial=digest(head)
    normalizations={k:fit_normalization(bank,rows,k,config["whitening_shrinkage"]) for k in ("zscore","whiten")}
    np.savez_compressed(dest/"train_normalizations.npz",**{k+"_"+side+"_"+field:value for k,v in normalizations.items() for side,x in v.items() for field,value in x.items() if field!="samples"})
    cost={label:geometry.evaluate(head,bank,rows,co,dest/("cost_"+label),config,normalizations) for label,co in cohort.items()}
    enc=np.load(clock/"encodings.npz")
    token_geometry=geometry.geometry(bank,rows,events,head,config);token_geometry["repeat_consistency"]=geometry.repeat_consistency(rows,enc,config["repeat_pairs"])
    token_geometry["global256_source_window_baseline"]={k:global_baseline(enc,rows,v) for k,v in cohort.items()}
    write(dest/"token_geometry.json",token_geometry)
    info=information.run(source,clock,rows,config,dest/"information")
    f_head,f_payload=frozen_head(source/"path_weak/head.pt")
    gradients=gradient.run(f_head,bank,rows,load_dense(clock),source,f_payload["config"],config,dest/"gradients")
    if digest(head)!=initial:raise ValueError("Read-only diagnosis changed source projection")
    conditional=dict(np.load(out/"conditional.npz"));runtime_protocol=dict(protocol,_original_out=str(out))
    if replay_out is not None:
        def forbid(*a,**k):raise RuntimeError("Replay optimizer forbidden")
        torch.optim.AdamW=forbid;torch.optim.Adam=forbid;torch.optim.SGD=forbid
    r2_result=r2(source,rows,bank,conditional,runtime_protocol,cohort,dest/"r2",out/"scientific/r2" if replay_out is not None else None)
    result=dict(status="COMPLETED_LOCAL_IDENTIFIABILITY_SHADOW",cost=cost,geometry=token_geometry,information=info,gradients=gradients,r2=r2_result,
        strict_population=protocol["strict_population"],encoder_changed=False,default_generation_changed=False,test_evaluations=0,
        true_event_semantics="UNKNOWN_NO_INDEPENDENT_LABELS",new_blind_source=False,operational_qualified=False)
    write(dest/"result.json",result)
    if replay_out is not None:
        # R2 training update accounting is execution bookkeeping, not a scientific measurement.
        if r2_result["status"]=="COMPLETED_EQUAL_CAPACITY_R2_SHADOW":
            original=read(out/"scientific/r2/result.json");r2_result["optimizer_updates"]=original["optimizer_updates"];write(dest/"r2/result.json",r2_result);result["r2"]=r2_result;write(dest/"result.json",result)
        fields=compare_tree(out/"scientific",dest)
        write(Path(replay_out)/"result.json",dict(status="FRESH_LOCAL_IDENTIFIABILITY_REPLAY_VERIFIED",execution_commit=protocol["code_commit"],
            replay_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),scientific_array_fields_verified=fields,
            optimizer_calls=0,diagnostic_virtual_clone_updates=gradients["diagnostic_virtual_clone_updates"],closed_form_probe_fits=10,operational_qualified=False))
        write(out/"replay_verified.json",ref(Path(replay_out)/"result.json"))
        print("LOCAL_REPLAY_VERIFIED",fields,flush=True)
    else:print("LOCAL_IDENTIFIABILITY_COMPLETED",flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=("preflight","run","replay"),required=True)
    for k in ("source","clock","config","replay-out"):p.add_argument("--"+k)
    p.add_argument("--out",required=True);a=p.parse_args()
    if a.phase=="preflight":preflight(a.source,a.clock,read(a.config),a.out)
    else:run(a.out,a.replay_out if a.phase=="replay" else None)
if __name__=="__main__":main()
