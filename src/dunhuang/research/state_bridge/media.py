"""Offline self-contained state inspector; old motion only, no smoothing or new poses."""
import argparse,json,os
from pathlib import Path
import numpy as np
from dunhuang.io import write
from dunhuang.data.sources import ref
from dunhuang.contracts import matrices
from dunhuang.research.source_capacity.common import read,commit,fresh
from .common import cases,native_environment
from .metrics import rotvec,FEET

def run(cfg,audit,analysis,out):
    out=fresh(out);items=cases(cfg);native_environment(items[0]["request"])
    from motion.geometry import fk_24_np
    audited=read(Path(audit)/"result.json");analyzed=read(Path(analysis)/"result.json")
    data=dict(cases={},training_admitted=analyzed["qualification"]["training_admitted"],
        audit_link=Path(os.path.relpath(Path(audit).resolve()/"result.json",out)).as_posix(),
        analysis_link=Path(os.path.relpath(Path(analysis).resolve()/"result.json",out)).as_posix())
    for item in items:
        x=item["original"][:150];blocks={}
        for arm in ("original","zero","C1","C2"):
            if arm=="original":native=x
            else:
                folder=item["parent_folder"] if arm=="C2" else Path(cfg["comparison"])/"generation"/(item["name"]+"_"+arm)
                native=np.load(folder/"edited_native.npy")[:150]
            fk=fk_24_np(native);R=matrices(native[:,7:13]);angle=np.linalg.norm(rotvec(R[:-1].swapaxes(-1,-2)@R[1:]),axis=-1)
            speed=np.linalg.norm(np.diff(fk[:,FEET][:,:,[0,2]],axis=0)*30,axis=-1).mean(-1)
            blocks[arm]=dict(fk=np.round(fk,6).tolist(),root=native[:,4:7].tolist(),
                root_step=[None]+np.linalg.norm(np.diff(native[:,4:7].astype(float),axis=0),axis=-1).tolist(),
                root_angle=[None]+angle.tolist(),foot_speed=[None]+speed.tolist(),
                native_ok=next(r["native_ok"] for r in audited["records"] if r["case"]==item["name"] and r["arm"]==arm))
        curve={a:[{k:r[k] for k in ("step","t","mse","root_rmse_m","pose_error_rad","feet_rmse_m")} for r in analyzed["records"] if r["case"]==item["name"] and r["arm"]==a] for a in cfg["arms"]}
        data["cases"][item["name"]]=dict(kind="AIST++真实配对目标" if item["name"].startswith("val_") else "历史R8冻结生成上下文",
            arms=blocks,floor=float(np.percentile(np.asarray(blocks["original"]["fk"])[:,FEET,1],5)),curves=curve)
    blob=json.dumps(data,ensure_ascii=False,separators=(",",":")).replace("<",chr(92)+"u003c")
    template=Path(__file__).with_name("viewer.html").read_text(encoding="utf8")
    (out/"index.html").write_text(template.replace("__DATA__",blob),encoding="utf8")
    checks=dict(cases=len(data["cases"]),arms=4,frames=150,world_root_preserved=True,fk_display_rounding_decimals=6,
        physical_metrics_use_original_full_precision=True,network_dependencies=0,new_neural_generations=0,human_reviewed=False,operational_qualified=False)
    write(out/"checks.json",checks);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,audit=ref(Path(audit)/"result.json"),analysis=ref(Path(analysis)/"result.json"),
        html=ref(out/"index.html"),checks=checks))
    print("R11_STATE_INSPECTOR",len(data["cases"]),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--audit",required=True);p.add_argument("--analysis",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(read(a.config),a.audit,a.analysis,a.out)
