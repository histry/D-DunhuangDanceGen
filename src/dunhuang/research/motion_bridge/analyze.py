"""Independent Native/SO3/FK readout. Qualification does not change Native limits."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.contracts import MotionAdapter,matrices
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.state_bridge.common import cases,native_environment
from dunhuang.research.condition_dependency.codec import restore
from dunhuang.research.bridge_inpainting.study import audit_native,region,source_error
from dunhuang.research.state_bridge.metrics import seam

def run(cfg,generation,out,replay=None):
    out=fresh(out);oldcfg=read("src/dunhuang/research/state_bridge/config.json");items=cases(oldcfg);native_environment(items[0]["request"])
    generated=read(Path(generation)/"result.json");records=[];controls=[]
    for i,item in enumerate(items):
        old=item["original"];oldfk,oldphysical=audit_native(old);q=item["request"];s=np.load(checked(q["scaler"]))
        for arm in ("original","B0","B1","B3","B2"):
            if arm=="original":x=old
            elif arm=="B1":x=np.load(item["parent_folder"]/"edited_native.npy")
            else:
                b=next(b for b in generated["bindings"] if b["case"]==item["name"] and b["arm"]==arm);x=np.load(checked(b["native"]))
            fk,physical=audit_native(x);_,roi=audit_native(x[:150])
            folder=out/(item["name"]+"_"+arm);folder.mkdir();np.save(folder/"fk.npy",fk);write(folder/"physical.json",physical);write(folder/"roi_physical.json",roi)
            assert np.array_equal(x[:60],old[:60]) and np.array_equal(x[90:],old[90:])
            R=matrices(x[60:90,7:].reshape(-1,24,6));G=matrices(old[60:90,7:].reshape(-1,24,6))
            pose=float(Rotation.from_matrix((G.swapaxes(-1,-2)@R).reshape(-1,3,3)).magnitude().mean())
            row=dict(case=item["name"],arm=arm,root_rmse_m=float(np.sqrt(np.mean((x[60:90,4:7]-old[60:90,4:7])**2))),
                pose_error_rad=pose,feet_rmse_m=float(np.sqrt(np.mean((fk[60:90,[7,8,10,11]]-oldfk[60:90,[7,8,10,11]])**2))),
                seams=[seam(x,fk,t,2e-5) for t in (60,90)],joins=[region(fk,t-3,t+3) for t in (60,90)],
                native_ok=physical["absolute"]["ok"],roi_native_ok=roi["absolute"]["ok"],reasons=physical["absolute"]["reasons"])
            if i>=4:row["source_error_rad"]=source_error(x,read(checked(item["binding"]["target"]["event_plan"])),60,90)
            if arm in ("B0","B3","B2"):
                p=Path(checked(b["raw_last"]));raw=np.load(p)
                raw_native=restore(MotionAdapter.to_native(((raw-s["bias"])/s["scale"]).astype(np.float32)),np.asarray(q["root_origin_native"],np.float32))
                row["raw_known_root_error_m"]=float(np.linalg.norm(raw_native[[9,40],4:7]-old[[59,90],4:7],axis=-1).mean())
                info=read(checked(b["trace_info"]))
                if i<4:
                    for control in info[0]["controls"]:
                        c=control["control"]
                        means={key:float(np.mean([next(v for v in t["controls"] if v["control"]==c)[key] for t in info])) for key in ("reconstruction","anchor_root","anchor_pose","seam_velocity","seam_acceleration","seam_rotation")}
                        controls.append(dict(case=item["name"],arm=arm,control=c,**means))
            records.append(row)
            if replay:
                other=Path(replay)/folder.name;assert np.array_equal(fk,np.load(other/"fk.npy"))
                assert physical==read(other/"physical.json") and roi==read(other/"roi_physical.json")
    comparisons={}
    for arm in ("B3","B2"):
        wins={};rootwins={}
        for control in ("removed","removed_left","removed_right","wrong_left","wrong_right"):
            n=0;nr=0
            for item in items[:4]:
                rs=[r for r in controls if r["case"]==item["name"] and r["arm"]==arm]
                correct=next(r for r in rs if r["control"]=="correct");cf=next(r for r in rs if r["control"]==control)
                n+=correct["reconstruction"]<cf["reconstruction"]
                nr+=correct["anchor_root"]<cf["anchor_root"]
            wins[control]=n;rootwins[control]=nr
        rows=[r for r in records if r["arm"]==arm]
        baseline=[next(r for r in records if r["case"]==x["case"] and r["arm"]=="B1") for x in rows]
        state_pass=all(wins[c]>=3 and rootwins[c]>=3 for c in ("wrong_left","wrong_right"))
        physical_pass=all(r["native_ok"] and r["roi_native_ok"] for r in rows)
        fidelity=all(r["pose_error_rad"]<=b["pose_error_rad"]+.01 for r,b in zip(rows,baseline))
        seams_pass=all(all(j["maximum"]<=k["maximum"]+1e-6 for j,k in zip(r["joins"],b["joins"])) for r,b in zip(rows,baseline))
        source_pass=all(r["source_error_rad"]<=b["source_error_rad"]+.01 for r,b in zip(rows,baseline) if "source_error_rad" in r)
        comparisons[arm]=dict(correct_beats_control_out_of_four=wins,root_correct_beats_control_out_of_four=rootwins,bidirectional_response_pass=state_pass,
            native_pass=physical_pass,fidelity_nonregression=fidelity,join_nonregression=seams_pass,source_nonregression=source_pass,
            single_window_qualified=state_pass and physical_pass and fidelity and seams_pass and source_pass)
    result=dict(records=records,controls=controls,qualification=comparisons,optimizer_steps=0,new_whole_song_generations=0,operational_qualified=False,
        interpretation="Shared-state controls test incremental soft mechanism, not complete hard-context causality. Four exposed Val cases are not independent generalization proof.")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,generation=ref(Path(generation)/"result.json"),original_native_protocol_unchanged=True))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R12_NATIVE",comparisons,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--generation",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.generation,a.out,a.replay_from)
