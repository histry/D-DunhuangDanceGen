"""Fresh saved-science verification: no fitting, optimizer or sampler."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read,commit,forbid_optimizers,same_arrays
from dunhuang.research.boundary_state_r14.bounds import immutable_jerk_bound
from dunhuang.research.boundary_state_r14.readout import qualification
from dunhuang.research.boundary_state_r14.states import FEATURE_DIM

def run(root):
    root=Path(root);forbid_optimizers();arrays=0
    for phase in ("audit","conflict","turn"):
        r=read(root/phase/"result.json");assert r==read(root/(phase+"_replay")/"result.json")
        assert r["optimizer_steps"]==0
    for f in ("originals.npz","skeleton.npz","train_parity.npz"):
        same_arrays(root/"audit"/f,root/"audit_replay"/f);arrays+=1
    from dunhuang.research.boundary_state_r14.common import population
    from dunhuang.research.state_bridge.common import native_environment
    cfg=read(root/"audit/protocol.json")["config"];assets,_,_,_,_=population(cfg);native_environment(assets)
    from contracts.physical_quality import PhysicalQualityLimits
    limit=PhysicalQualityLimits.from_environment().joint_jerk_mps3_max
    audit=read(root/"audit/result.json");saved=np.load(root/"audit/originals.npz")
    assert len(audit["records"])==400 and {r["case"] for r in audit["records"]}=={0,1,2,3}
    for b in audit["necessary_bounds"]:
        bound=immutable_jerk_bound(saved["fk_"+str(b["case"])],limit,b["edit"],b["fps"])
        for k,v in bound.items():assert v==b[k]
    assert read(root/"conflict/result.json")["finite_parameter_interventions_restored_exact"]
    a=read(root/"readout/result.json");b=read(root/"readout_replay/result.json")
    cfg=read(root/"readout/protocol.json")["config"]
    assert qualification(a["group_scores"],cfg)==a["qualification"]
    for key in ("records","group_scores","qualification","selected_family","parameters_per_head"):assert a[key]==b[key]
    assert a["analytical_heads_fitted"]==5 and b["analytical_heads_fitted"]==0
    assert a["parameters_per_head"]==FEATURE_DIM*270+270
    for f in cfg["head_families"]:
        same_arrays(root/"readout"/(f+".npz"),root/"readout_replay"/(f+".npz"));arrays+=1
        m=np.load(checked(a["models"][f]));assert m["coef"].shape==(FEATURE_DIM,270)
    same_arrays(root/"readout/predictions.npz",root/"readout_replay/predictions.npz");arrays+=1
    p=read(root/"readout/protocol.json");fit=p["fit"];hold=p["holdout"];rows=fit+hold
    fit_groups={r["source_group"] for r in fit};hold_groups={r["source_group"] for r in hold}
    assert len(fit_groups)==28 and len(hold_groups)==8 and not fit_groups&hold_groups
    for i,j in enumerate(p["donor_indices"]):
        assert (i<len(fit))==(j<len(fit)) and rows[i]["source_group"]!=rows[j]["source_group"]
    assert len(set(p["donor_indices"]))==len(rows)
    c=read(root/"capacity/result.json");cr=read(root/"capacity_replay/result.json")
    assert c["derivatives"]==cr["derivatives"] and c["model_unchanged"] and c["optimizer_steps"]==0
    assert all(x["row"]["source_group"] in hold_groups for x in c["bindings"])
    for binding in c["bindings"]:
        name="case_"+str(binding["case"])
        for f in ("trace.npz","probes.npz"):
            same_arrays(root/"capacity"/name/f,root/"capacity_replay"/name/f);arrays+=1
        assert read(root/"capacity"/name/"sampler.json")==read(root/"capacity_replay"/name/"sampler.json")
    same_arrays(root/"capacity/references.npz",root/"capacity_replay/references.npz");arrays+=1
    n=read(root/"capacity_native/result.json");assert n==read(root/"capacity_native_replay/result.json")
    for folder in (root/"capacity_native").iterdir():
        if not folder.is_dir():continue
        for f in ("native.npy","fk.npy"):
            assert np.array_equal(np.load(folder/f),np.load(root/"capacity_native_replay"/folder.name/f));arrays+=1
        assert read(folder/"physical.json")==read(root/"capacity_native_replay"/folder.name/"physical.json")
    for x in read(root/"turn/protocol.json")["trace_inputs"]:checked(x)
    frozen=subprocess.check_output(["git","diff","--name-only","e0b7100","--","src/dunhuang","configs","tests/test_boundary_state.py","reports/r13_summary.json","reports/r13_checks.json","reports/r13_receipt.json"],text=True).splitlines()
    frozen=[f for f in frozen if not f.startswith("src/dunhuang/research/boundary_state_r14/")];assert not frozen
    receipts={}
    for stage in ("audit","readout","capacity"):
        path=Path(str(root)+"_"+stage+"_receipts")
        assert (path/"exit_status.txt").read_text().strip()=="0"
        receipts[stage]=(path/"execution_commit.txt").read_text().strip()
    result=dict(code_commit=commit(),science_arrays_exact=arrays,frozen_changed_paths=frozen,
        frozen_audit_states=400,train_conflict_states=16,readout_fit_windows=len(fit),readout_holdout_windows=len(hold),
        group_roles_disjoint=True,donors_same_role_cross_group=True,readout_models_exact=5,
        capacity_baseline_rollouts_exact=8,capacity_state_readouts_exact=len(n["records"]),
        analytical_heads_fitted=0,replay_optimizer_steps=0,optimizer_steps=0,model_weights_restored=True,
        new_whole_song_generations=0,operational_qualified=False,receipts=receipts)
    write(root/"independent_checks.json",result);print("R14_VERIFIED",result,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
