"""Independent final readout; verify frozen inputs, marginals, Native and role controls."""
import argparse,collections,json,re,subprocess,sys,ast
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import checked,sha,write
from dunhuang.research.source_identifiability.a2 import candidates
from dunhuang.generation.event_music_adapter import authorize_event_adapter

def read(p):return json.loads(Path(p).read_text())
def seal(out,receipt):
    out=Path(out);receipt=Path(receipt);destination=out/"public_receipt.json"
    if destination.exists():raise ValueError("Receipt destination exists")
    if (receipt/"exit_status.txt").read_text().strip()!="0":raise ValueError("Incomplete tmux execution")
    public=read(out/"public_summary.json");g=public["g2"];a=public["a2"]
    for name,h in public["evidence"].items():
        if sha(out/name)!=h:raise ValueError("Scientific export seal differs")
    baseline="1e8f99b0ff08266968b529615b0917028b493e6a"
    frozen=["src/dunhuang/edge_backend.py","src/dunhuang/sampling.py","src/dunhuang/physical.py","src/dunhuang/contracts.py",
        "src/dunhuang/alignment/model.py","src/dunhuang/generation/event_geometry.py","src/dunhuang/generation/event_visibility.py","src/dunhuang/research/frozen.py"]
    if subprocess.check_output(["git","diff","--name-only",baseline,"HEAD","--"]+frozen,text=True).strip():raise ValueError("Frozen baseline numerical implementation changed")
    def original(path):return subprocess.check_output(["git","show",baseline+":"+path],text=True)
    adapter="src/dunhuang/generation/event_music_adapter.py"
    old=next(n for n in ast.parse(original(adapter)).body if isinstance(n,ast.ClassDef) and n.name=="EventMusicAdapter")
    new=next(n for n in ast.parse(Path(adapter).read_text()).body if isinstance(n,ast.ClassDef) and n.name=="EventMusicAdapter")
    if ast.dump(old)!=ast.dump(new):raise ValueError("Adapter tensor architecture changed")
    original_targets={n.name:ast.dump(n) for n in ast.parse(original("src/dunhuang/generation/train.py")).body if isinstance(n,ast.FunctionDef)}
    current_targets={n.name:ast.dump(n) for n in ast.parse(Path("src/dunhuang/generation/targets.py").read_text()).body if isinstance(n,ast.FunctionDef)}
    if any(original_targets[name]!=current_targets[name] for name in ("phase_target","digest")):raise ValueError("Shared target extraction changed mathematics")
    p=read(out/"g2/protocol.json");manifest=read(checked(p["paired_manifest"]));index={r["window_uid"]:r for r in manifest["rows"]}
    initial=torch.load(checked(p["initial_adapter"]),map_location="cpu",weights_only=True)
    authorize_event_adapter(initial,p["base"]["sha256"],True)
    schedule=np.load(out/"g2/training_schedule.npy");donor=np.load(out/"g2/donor_schedule.npy")
    if not np.array_equal(np.sort(schedule.flatten()),np.sort(donor.flatten())):raise ValueError("Corruption source marginal differs")
    for x,y in zip(schedule.flatten(),donor.flatten()):
        left=index[p["train_uids"][int(x)]];right=index[p["train_uids"][int(y)]]
        if left["role"]!="train" or right["role"]!="train" or left["source_group"]==right["source_group"] or left["choreography_uid"]==right["choreography_uid"]:raise ValueError("Source identity control leaked")
    model_parameters={}
    for arm in ("g2_1","g2_2","g2_3"):
        if sha(out/"g2"/arm/"adapter.pt")!=public["model_hashes"][arm]:raise ValueError("Trained model seal differs")
        payload=torch.load(out/"g2"/arm/"adapter.pt",map_location="cpu",weights_only=True)
        authorize_event_adapter(payload,p["base"]["sha256"],True,visibility_policy="observed")
        if payload["selected_step"]!=400 or payload["protocol_sha256"]!=sha(out/"g2/protocol.json"):raise ValueError("G2 checkpoint selection differs")
        if set(payload["state"])!=set(initial["state"]):raise ValueError("G2 architecture differs")
        model_parameters[arm]=sum(v.numel() for v in payload["state"].values())
    if len(set(model_parameters.values()))!=1:raise ValueError("G2 parameter budgets differ")
    arm_summary={}
    for arm in ("g2_0","g2_1","g2_2","g2_3","hard","interpolation"):
        prefs=[r for r in g["preferences"] if r["arm"]==arm]
        rows=[r for r in g["records"] if r["control"] in (arm+"_0",arm+"_1")]
        arm_summary[arm]=dict(cases=len(prefs),following_case_passes=sum(r["source_following_pass"] for r in prefs),
            minimum_directional_preference=min(min(r["source_a_preference"],r["source_b_preference"]) for r in prefs),
            mean_preference=float(np.mean([r["mean_preference"] for r in prefs])),
            mean_reference_gain_vs_bypass=float(np.mean([r["reference_gain_vs_bypass"] for r in prefs])),
            minimum_activity_retention=min(r["activity_retention"] for r in prefs),
            provided_error_mean_rad=float(np.mean([r["known_error_radians"] for r in rows])),
            unprovided_error_mean_rad=float(np.mean([r["unprovided_error_radians"] for r in rows])),
            key_error_mean_rad=float(np.mean([r["key_error_radians"] for r in rows])),
            native_passes=sum(r["native_ok"] for r in rows),native_samples=len(rows),
            root_speed_mean_mps=float(np.mean([r["dynamics"]["root_speed_mean_mps"] for r in rows])),
            jerk_p95_mean_mps3=float(np.mean([r["dynamics"]["jerk_p95_mps3"] for r in rows])))
    failures=collections.Counter();limits=None
    physical_files=list((out/"g2").glob("case_*/**/physical.json"))
    if len(physical_files)!=52:raise ValueError("Native artifact population incomplete")
    for file in physical_files:
        physical=read(file);replay=read(out/"g2_replay"/file.relative_to(out/"g2"))
        if physical!=replay:raise ValueError("Full Native audit changed on replay")
        failures.update(physical["absolute"]["reasons"])
        if limits is None:limits=physical["limits"]
        elif physical["limits"]!=limits:raise ValueError("Physical limits differ across controls")
    ap=read(out/"a2/protocol.json");rows=read(checked(ap["source_manifest"]))["rows"]
    nodes=dict(np.load(checked(ap["source_nodes"])));stats=dict(np.load(out/"a2/nuisance.npz"));saved=dict(np.load(out/"a2/candidates.npz"))
    fresh=dict(np.load(out/"a2_replay/nuisance.npz"))
    if any(not np.array_equal(v,fresh[k]) for k,v in stats.items()):raise ValueError("Rebuilt nuisance observations differ")
    v2_intersections={}
    for role in ("train","val"):
        ids=np.array([i for i,r in enumerate(rows) if r["role"]==role])
        current,e=candidates([rows[i] for i in ids],{k:nodes[k][ids] for k in ("mv","dv")},{k:v[ids] for k,v in stats.items()},ap["config"]["strict"])
        if not np.array_equal(ids[current["candidates"]],saved["candidates"][ids]) or any(not np.array_equal(current[k],saved[k][ids]) for k in ("positive","valid","eligible")):raise ValueError("Role-specific candidate replay differs")
        v2_intersections[role]=e["rule_failure_intersections"]
    counts=lambda name:(re.findall(r"\d+ passed(?:, \d+ skipped)?",(receipt/name).read_text()) or [None])[-1]
    result=dict(schema="g2_a2_completed_receipt_v1",execution_commit=public["execution_commit"],
        readout_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),exit_status=0,
        started_utc=(receipt/"started_utc.txt").read_text().strip(),finished_utc=(receipt/"finished_utc.txt").read_text().strip(),
        production_edge_sampling_native_encoder_unchanged=True,adapter_tensor_implementation_unchanged=True,shared_target_functions_unchanged=True,
        nuisance_sha256=sha(out/"a2/nuisance.npz"),exact_source_marginal_verified=True,corrupted_source_pairs=int(schedule.size),matched_model_parameters=model_parameters,
        optimizer_updates=3*400,checkpoint_selection="fixed_last400_no_Val_selection",source_following=arm_summary,
        diffusion_generations=g["generations"],nonlearned_interpolations=g["nonlearned_interpolations"],
        fresh_diffusion_regenerations=public["replay"]["g2"]["regenerated"],native_passes=g["native_passes"],native_evaluations=g["native_evaluations"],
        native_failure_histogram=dict(failures),native_limits_constant=True,
        a2_population=a["populations"],a2_clock=a["clock"],a2_v2_failures_by_role=v2_intersections,
        a2_optimizer_updates=a["training"]["optimizer_updates"],a2_role_candidate_replay_verified=True,
        final_validation={name:counts(name) for name in ("validation_pytest.log","validation_edge.log")},
        logs_sha256={name:sha(receipt/name) for name in ("pytest.log","edge_check.log","validation_pytest.log","validation_edge.log","execution_commit.txt","exit_status.txt","started_utc.txt","finished_utc.txt")},
        public_summary_sha256=sha(out/"public_summary.json"),scientific_artifacts_unchanged=True,new_blind_source=False,human_reviewed=False,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,g3_started=False,sqp_started=False,operational_qualified=False)
    if not all(result["final_validation"].values()):raise ValueError("Final code validation absent")
    if any(s in json.dumps(result) for s in ("/home/","window_uid","aist_capture:","known_component:","smpl_poses")):raise ValueError("Private receipt rejected")
    write(destination,result);print("SOURCE_IDENTIFIABILITY_RECEIPT_SEALED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--receipt",required=True)
    a=p.parse_args();seal(a.out,a.receipt)
