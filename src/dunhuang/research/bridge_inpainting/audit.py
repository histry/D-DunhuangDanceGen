"""R8-B0: locate the earliest AVAILABLE motion stage, audit fixed contexts."""
import argparse,copy,sys
from pathlib import Path
import numpy as np
from dunhuang.io import checked,sha,write,pinned
from dunhuang.data.sources import ref
from dunhuang.contracts import MotionAdapter,matrices,Q
from dunhuang.research.source_capacity.common import read,commit,fresh

def distribution(x):
    x=np.asarray(x,float);return dict(p95=float(np.percentile(x,95)),maximum=float(x.max()))
def context_binding(native,scale,bias,cfg):
    values=MotionAdapter.to_edge(native[:150]);mask=np.zeros((150,151),bool)
    for a,b in (cfg["left_context"],cfg["right_context"]):mask[a:b,4:]=True
    affine=values*scale+bias;outside=mask&(abs(affine)>1+32*np.finfo(np.float32).eps)
    return values[None],mask[None],int(outside.sum())
def run(cfg,out):
    out=fresh(out);parent=Path(cfg["prior"]);pub=read("reports/exposed_full_song_summary.json")
    for phase in ("music","features","generation"):
        if sha(parent/phase/"protocol.json")!=pub["evidence"][phase+"/protocol.json"]:raise ValueError("Frozen whole-song source seal differs")
    assets=read(cfg["assets"]);sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    from motion.geometry import fk_24_np
    protocol=read(parent/"generation/protocol.json");scaler=np.load(checked(assets["scaler"]));rows=[];candidates=[]
    mp=read(parent/"music/protocol.json");plans=read(checked(mp["generation_plans"]));features=read(parent/"features/protocol.json")
    side=read(Path(cfg["data_v3"])/"main/event_catalog_v2_shadow.json")["entries"];index={e["event_uid"]:e for e in side}
    for binding in protocol["bindings"]:
        folder=Path(checked(binding["native"])).parent;name=folder.name;native=np.load(folder/"native.npy")
        windows=np.load(folder/"windows.npy");edge=np.load(folder/"edge.npy");official=np.load(folder/"official_fk.npy")[:len(native)]@Q
        t=np.arange(len(native));w=np.minimum(t//75,len(windows)-1);k=t-75*w
        before=MotionAdapter.to_native(windows[w,k].astype(np.float32))
        root_delta=float(abs(before[:,4:7]-native[:,4:7]).max())
        a=matrices(before[:,7:].reshape(-1,24,6));b=matrices(native[:,7:].reshape(-1,24,6))
        p=fk_24_np(before);actual=np.load(folder/"native_fk.npy")
        rows.append(dict(name=name,frames=len(native),raw_window_root_jerk_mps3=distribution(np.linalg.norm(np.diff(before[:,4:7],n=3,axis=0)*30**3,axis=-1)),
            assembled_root_jerk_mps3=distribution(np.linalg.norm(np.diff(native[:,4:7],n=3,axis=0)*30**3,axis=-1)),
            root_assembly_max_abs_m=root_delta,raw_window_vs_assembled_fk_max_abs_m=float(abs(p-actual).max()),native_vs_official_fk_max_abs_m=float(abs(actual-official).max()),
            overlap_raw_max_abs=float(abs(windows[:-1,75:]-windows[1:,:75]).max()),
            earliest_available_stage="raw denormalized denoising windows; no per-step trajectory saved",causal_claim=False))
        if name not in cfg["cases"]:continue
        i=int(name.split("_")[0]);original_plan=read(checked(plans[i]["plans"]["A"]))
        for e in original_plan["events"]:
            metadata=index[e["event_uid"]];e.update(observed_joints=metadata["observed_joints"],full_phases=metadata["kinematic_phases"],key_motion=metadata["key_motion"],root_translation_observed=False)
        file=out/(name+"_plan.json");write(file,original_plan)
        values,mask,unsupported=context_binding(native,scaler["scale"],scaler["bias"],cfg)
        supported=unsupported==0;context=out/(name+"_context.npz");np.savez_compressed(context,values=values,mask=mask)
        music=out/(name+"_music.npy");np.save(music,np.load(checked(features["songs"][i]["features"]))[:1])
        candidates.append(dict(name=name,case=i,native=binding["native"],fk=binding["fk"],original_physical=binding["physical"],original_request=binding["request"],
            context=ref(context),context_supported=supported,unsupported_coordinates=unsupported,context_root_observed_as_generated_data=True,context_is_source_truth=False,
            features=ref(music),audio=features["songs"][i]["audio"],event_plan=ref(file),source_row=ref(parent/"generation/protocol.json")))
    assert len(rows)==10 and len(candidates)==len(cfg["cases"])
    write(out/"result.json",dict(rows=rows,candidate_count=len(candidates),supported=sum(c["context_supported"] for c in candidates),new_generation_calls=0,neural_weight_updates=0))
    write(out/"candidates.json",candidates);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,assets=ref(cfg["assets"]),parent=ref(parent/"generation/protocol.json"),
        candidates=ref(out/"candidates.json"),phase_catalog=ref(Path(cfg["data_v3"])/"main/event_catalog_v2_shadow.json"),
        policy="fixed first 75-frame boundary; no replacement after support or outcome failure; canonical actual generated context is never clipped"))
    print("R8_BOUNDARY_AUDIT",len(candidates),sum(c["context_supported"] for c in candidates),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(read(a.config),a.out)
