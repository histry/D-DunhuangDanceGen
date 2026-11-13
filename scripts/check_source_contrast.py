"""Independent R7 saved-output/source-gain and coefficient-domain checks."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,sha,write
from dunhuang.contracts import MotionAdapter,matrices
def read(p):return json.loads(Path(p).read_text())
def angle(a,b,mask):
    x=matrices(a[:,79:127].reshape(150,8,6)).astype(float);y=matrices(b[:,79:127].reshape(150,8,6)).astype(float)
    return float(Rotation.from_matrix((x.swapaxes(-1,-2)@y)[mask]).magnitude().mean())
def check(root):
    root=Path(root);pre=read(root/"preflight/protocol.json");config=pre["config"];cfg=read(config["frozen_config"])
    pairs=read(checked(pre["pairs"]));manifest=read(checked(pre["source_manifest"]));index={r["window_uid"]:r for r in manifest["rows"]}
    from dunhuang.research.source_capacity.common import frozen
    _,_,p,_=frozen(cfg);assets=read(checked(p["assets"]));scaler=np.load(checked(assets["scaler"]));checked(assets["checkpoint"])
    arrays=dict(np.load(root/"control/arrays.npz"));replay=np.load(root/"control_replay/arrays.npz")
    assert set(arrays)==set(replay.files)
    for k,v in arrays.items():assert np.array_equal(v,replay[k])
    result=read(root/"control/result.json");assert result==read(root/"control_replay/result.json")
    old=np.load(checked(pre["frozen_probe_arrays"]));mask=np.ones((150,8),bool);mask[25:40]=False;mask[90:115]=False
    def native(value):return MotionAdapter.to_native(((value[0].clip(-1,1)-scaler["bias"])/scaler["scale"]).astype(np.float32))
    from dunhuang.research.reference_guidance.core import protection
    count=0;maxerror=0.;dose={};budget=0.;joint={};small=0;oldparities=0;physicalerror=0.
    solvers={(s["cohort"],s["case"],s["timestep"],s["source"],s["arm"]):s for s in result["solves"]}
    for r in result["records"]:
        cohort,case,t,s,arm=(r[k] for k in ("cohort","case","timestep","source","arm"));prefix=cohort+"__"+str(case)+"__"+str(t);key=prefix+"__"+str(s)+"__"+arm+"__"+r["dose_kind"]
        base=native(arrays[prefix+"__base"]);output=native(arrays[key+"__predicted_x0"]);pair=pairs[cohort][case]
        refs=[np.load(checked(index[pair[k]]["motion"])) for k in ("a","b")]
        own=angle(base,refs[s],mask)-angle(output,refs[s],mask);other=angle(base,refs[1-s],mask)-angle(output,refs[1-s],mask)
        maxerror=max(maxerror,abs(own-r["own_gain_rad"]),abs(other-r["other_gain_rad"]),abs(own-other-r["directional_gain_rad"]))
        p=protection(base,output,cfg["protection"]);assert p["protected"]==r["protection"]["protected"]
        physicalerror=max(physicalerror,max(abs(p[k]-r["protection"][k]) for k in p if type(p[k]) is float))
        d=arrays[key+"__direction"];norm=float(np.linalg.norm(d.astype(float)));assert abs(norm-r["input_norm"])<1e-6;budget=max(budget,norm/r["budget_norm"])
        if r["dose_kind"]=="common_norm":
            dose.setdefault((cohort,case,t,s),[]).append(norm);assert abs(norm-r["common_norm"])<=2*np.finfo(np.float32).eps*max(1,r["common_norm"])
            ok=own>config["own_margin_rad"] and own-other>config["relative_margin_rad"] and p["protected"] and r["numeric_trusted"]
            joint[(cohort,arm)]=joint.get((cohort,arm),0)+int(ok)
        if r["dose_kind"]=="full_budget" and arm in ("A0","A4"):
            suffix="D0" if arm=="A0" else "D4";oldkey=prefix+"__"+str(s)+"__"+suffix+"__full_budget"
            assert np.array_equal(arrays[key+"__direction"],old[oldkey+"__direction"]) and np.array_equal(arrays[key+"__predicted_x0"],old[oldkey+"__predicted_x0"]);oldparities+=1
        count+=1
    for cohort in config["cohorts"]:
        for arm in config["arms"]:assert joint[(cohort,arm)]==result["qualification"][cohort][arm]["joint_trusted"]
    for solver in result["solves"]:
        if not solver["new_solve"]:continue
        prefix=solver["cohort"]+"__"+str(solver["case"])+"__"+str(solver["timestep"])+"__"+str(solver["source"])+"__"+solver["arm"]+"__small_"
        a=arrays[prefix+"physical"];g=arrays[prefix+"gram"];gain=arrays[prefix+"gains"];z=arrays[prefix+"coeff"]
        assert len(z)<=6 and a.shape==(8250,len(z)) and gain.shape==(2,len(z))
        if solver["status"]=="SOLVED_FINITE_SUBSPACE":
            v=a@z;contact=v[450:1050]*np.sqrt(600);other=v[1050:].reshape(2400,3)*np.sqrt(2400)
            assert np.linalg.norm(v[:450])<=.9+1e-6 and abs(contact).max()<=.9+1e-6 and np.linalg.norm(other,axis=-1).mean()<=.9+1e-6 and z@g@z<=1+1e-6
            assert np.all(gain@z>=np.array([config["own_margin_rad"],config["relative_margin_rad"]])-1e-6)
        else:assert not z.any()
        small+=1
    assert maxerror<1e-6 and physicalerror<1e-6 and budget<=1+1e-6 and count==480 and small==144 and oldparities==96
    from dunhuang.alignment.audio_clock import same_music_time
    assert all(same_music_time(index[p["a"]],index[p["b"]]) for pop in pairs.values() for p in pop)
    models=read("reports/source_tasks_summary.json")["model_sha256"]
    for name,digest in models.items():assert sha(Path(cfg["source"])/name/("adapter.pt" if name.startswith("cf/") else "content.pt"))==digest
    write(root/"public_checks.json",dict(code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),verifier_sha256=sha(__file__),
        saved_predictions_verified=count,exact_replay_arrays=len(arrays),source_readout_max_error_rad=maxerror,physical_readout_max_error=physicalerror,
        max_budget_fraction=budget,common_dose_groups=len(dose),zero_common_dose_groups=sum(max(v)==0 for v in dose.values()),small_projections_verified=small,frozen_prediction_parities=oldparities,
        frozen_models_verified=models,independent_evaluation="NOT_EVALUABLE_EXPOSED_TRAIN",optimizer_calls=0,operational_qualified=False))
    print("INDEPENDENT_R7_SOURCE_CHECKS",count,len(arrays),small)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args();check(a.out)
