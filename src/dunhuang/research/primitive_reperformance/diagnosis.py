"""R8-S0: raw ChangE, E1-W reference, and frozen R7 generation; no new generation."""
import argparse,sys,hashlib
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write,pinned
from dunhuang.data.sources import ref
from dunhuang.contracts import matrices
from dunhuang.generation.soft_reference import resample_native
from dunhuang.planning.time_map import source_positions
from dunhuang.research.source_capacity.common import read,commit,fresh

def canonical(points,native):
    r=matrices(native[:,7:13]);yaw=float(np.arctan2(r[0,0,2],r[0,2,2]));c,s=np.cos(yaw),np.sin(yaw)
    turn=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return ((points-points[:,:1])@turn).astype(np.float32)

def run(cfg,out):
    out=fresh(out);assets=read(cfg["assets"]);sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    from motion.geometry import fk_24_np
    prior=Path(cfg["prior_song"]);mp=read(prior/"music/protocol.json");gp=read(prior/"generation/protocol.json");plans=read(checked(mp["generation_plans"]))
    choices={}
    for item in plans:
        plan=read(checked(item["plans"]["A"]));case=item["case"]
        binding=next(b for b in gp["bindings"] if b["case"]==case and b["arm"]=="temporal_A")
        for e in plan["events"]:
            if not 75<=e["source_interval"][1]-e["source_interval"][0]<=150:continue
            identity=hashlib.sha256((str(cfg["seed"])+str(case)+e["event_uid"]).encode()).hexdigest()
            row=(identity,case,e,binding)
            if e["source_uid"] not in choices or identity<choices[e["source_uid"]][0]:choices[e["source_uid"]]=row
    records=[]
    for i,(_,case,e,binding) in enumerate(choices[k] for k in sorted(choices)):
        source=np.load(checked(e["reference"]));lo,hi=e["source_interval"];a,b=e["music_interval"]
        raw=source[lo:hi].copy();aligned=resample_native(source,source_positions(e,np.arange(a,b)));generated=np.load(checked(binding["native"]))[a:b]
        arrays=dict(raw=canonical(fk_24_np(raw),raw),retimed=canonical(fk_24_np(aligned),aligned),generated=canonical(fk_24_np(generated),generated))
        file=out/(str(i)+"_three_way.npz");np.savez_compressed(file,**arrays)
        records.append(dict(case=i,music_case=case,event_uid=e["event_uid"],source=e["reference"],source_interval=[lo,hi],music_interval=[a,b],time_warp=e["time_warp"],
            arrays=ref(file),source_observed_joints=list(range(22)),world_root_truth=False,fingers_observed=False,original_frames=len(raw),target_frames=len(aligned),cultural_truth="UNKNOWN"))
    write(out/"result.json",dict(records=records,new_generation_calls=0,neural_weight_updates=0,human_reviewed=False,semantic_style_qualified=False))
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,old_music=ref(prior/"music/protocol.json"),old_generation=ref(prior/"generation/protocol.json"),
        geometry="same Native FK24 and camera; only 22 fitted joints displayed; pelvis relative with initial heading canonicalized; raw source clock and target clock displayed separately"))
    print("R8_STYLE_THREE_WAY",len(records),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(read(a.config),a.out)
