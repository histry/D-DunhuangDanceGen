"""Locate first join jump: raw x0, hard mask, unnormalization, assembly, restore, graft."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.contracts import MotionAdapter,Q,matrices
from dunhuang.research.condition_dependency.codec import restore
from dunhuang.research.source_capacity.common import read,commit,fresh
from .common import cases,native_environment
from .metrics import FEET,rotvec

def run(cfg,trace,out,replay=None):
    out=fresh(out);items=cases(cfg);native_environment(items[0]["request"])
    from motion.geometry import fk_24_np
    p=read(Path(trace)/"protocol.json");records=[]
    for item,b in zip(items,p["bindings"]):
        folder=item["parent_folder"];q=item["request"];old=item["original"];o=np.asarray(q["root_origin_native"],np.float32)
        s=np.load(checked(q["scaler"]));norm=np.load(folder/"normalized.npy")[0]
        window=np.load(folder/"windows.npy")[0];edge=np.load(folder/"edge.npy")
        native=np.load(folder/"native.npy");sample_fk=np.load(folder/"native_fk.npy")+o
        edited=np.load(folder/"edited_native.npy");edited_fk=np.load(folder/"edited_fk.npy")
        with np.load(checked(b["trace"])) as z:raw=z["prediction"][-1,0]
        mask=np.load(checked(q["context"]))["mask"][0,50:100]
        hard=np.where(mask,norm[50:100],raw)
        assert np.array_equal(hard,norm[50:100])
        assert np.array_equal(raw[10:40],norm[60:90])
        raw_native=restore(MotionAdapter.to_native(((raw-s["bias"])/s["scale"]).astype(np.float32)),o)
        raw_fk=fk_24_np(raw_native)
        root_from_norm=((norm[:,4:7]-s["bias"][4:7])/s["scale"][4:7])
        checks=dict(hard_overwrite_exact=True,unknown_matches_raw_x0_exact=True,
            inverse_scaler_root_max_m=float(abs(root_from_norm-window[:,4:7]).max()),
            one_window_assembly_root_max_m=float(abs(window[:,4:7]-edge[:,4:7]).max()),
            native_coordinate_root_max_m=float(abs(edge[:,4:7]@Q-native[:,4:7]).max()),
            restored_to_graft_unknown_root_max_m=float(abs(native[60:90,4:7]+o-edited[60:90,4:7]).max()),
            restored_known_to_original_root_max_m=float(abs(native[[59,90],4:7]+o-old[[59,90],4:7]).max()))
        assert all(v<=cfg["oracle_fk_atol_m"] for k,v in checks.items() if k.endswith("_max_m"))
        seams=[]
        for t in (60,90):
            i=t-50
            def rootstep(x,index):return float(np.linalg.norm(x[index]-x[index-1]))
            def footstep(x,index):return float(np.linalg.norm(x[index,FEET]-x[index-1,FEET],axis=-1).mean())
            known=59 if t==60 else 90;ki=known-50
            raw_R=matrices(raw_native[ki,7:13]);gt_R=matrices(old[known,7:13])
            seams.append(dict(frame=t,original_root_step_m=rootstep(old[:,4:7],t),
                raw_model_root_step_m=rootstep(raw_native[:,4:7],i),
                raw_model_known_root_error_m=float(np.linalg.norm(raw_native[ki,4:7]-old[known,4:7])),
                raw_model_known_root_angle_error_rad=float(np.linalg.norm(rotvec(gt_R.T@raw_R))),
                after_hard_root_step_m=rootstep(root_from_norm@Q+o,t),
                after_unnormalization_root_step_m=rootstep(window[:,4:7]@Q+o,t),
                after_assembly_root_step_m=rootstep(edge[:,4:7]@Q+o,t),
                after_restore_root_step_m=rootstep(native[:,4:7]+o,t),
                after_graft_root_step_m=rootstep(edited[:,4:7],t),
                raw_model_feet_step_mean_m=footstep(raw_fk,i),
                after_hard_feet_step_mean_m=footstep(sample_fk,t),
                after_graft_feet_step_mean_m=footstep(edited_fk,t)))
        records.append(dict(case=item["name"],checks=checks,seams=seams))
    result=dict(records=records,optimizer_steps=0,new_neural_generations=0,operational_qualified=False,
        interpretation="known-state correction is a sampler hard-mask intervention; compatible motion must be predicted before forcing anchors; stage comparison is not an independent hard-path causal ablation")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,trace=ref(Path(trace)/"protocol.json")))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R11_STAGE_CHAIN",len(records),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--trace",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.trace,a.out,a.replay_from)
