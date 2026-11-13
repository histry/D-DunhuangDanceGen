"""Post-run diagnostics and complete source-gallery; no weight selection or sampling."""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,write,pinned,sha
from dunhuang.contracts import matrices
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.primitive_reperformance.metrics import angles
from dunhuang.research.primitive_reperformance.reference import build

def foot_factors(base,binding):
    x=np.load(checked(base["native"]));y=np.load(checked(binding["native"]))
    f0=Path(base["native"]["path"]).parent/"native_fk.npy";f1=Path(binding["native"]["path"]).parent/"native_fk.npy"
    p0=np.load(f0)[:,(7,8,10,11)].astype(float);p1=np.load(f1)[:,(7,8,10,11)].astype(float)
    r0=matrices(x[:,7:13]).astype(float);r1=matrices(y[:,7:13]).astype(float)
    l0=np.einsum("tji,tkj->tki",r0,p0-x[:,None,4:7]);l1=np.einsum("tji,tkj->tki",r1,p1-y[:,None,4:7])
    components=dict(root_translation=(y[:,None,4:7]-x[:,None,4:7]).astype(float),
        root_rotation=.5*np.einsum("tij,tkj->tki",r1-r0,l0+l1),
        local_pose=.5*np.einsum("tij,tkj->tki",r0+r1,l1-l0))
    error=float(abs(sum(components.values())-(p1-p0)).max());assert error<2e-5
    output={}
    for order,rate,label in ((1,30,"velocity"),(3,27000,"jerk")):
        total=np.diff(p1-p0,n=order,axis=0)*rate;power=float(np.sum(total**2));blocks={}
        for name,value in components.items():
            v=np.diff(value,n=order,axis=0)*rate
            blocks[name]=dict(rms=float(np.sqrt(np.mean(np.sum(v*v,axis=-1)))),signed_projection_share=float(np.sum(v*total)/power) if power>1e-16 else None)
        output[label]=blocks
    return dict(pair=binding["pair"],source=binding["source"],arm=binding["arm"],factors=output,position_identity_error_m=error,
        fk_sha256=[sha(f0),sha(f1)],contact_prediction_change_ratio=float(np.mean((x[:,:4]>=.5)!=(y[:,:4]>=.5))),
        interpretation="symmetric numerical decomposition of saved FK changes; signed shares can cancel, not physical causal or contact truth")

def run(main,out,render=True):
    out=fresh(out);main=Path(main);p=read(main/"source/protocol.json");result=read(main/"source/result.json");groups={}
    for arm in ("T0","T1","T2","T2R","T3","T4"):
        rows=[r for r in result["records"] if r["arm"]==arm];baseline={(r["pair"],r["source"]):r for r in result["records"] if r["arm"]=="T0"}
        group=dict(cases=len(rows),native_passes=sum(r["native_ok"] for r in rows))
        for k in ("mean_source_rad","phase_pose_rad","wrist_rotation_rad","trunk_rotation_rad","root_relative_turn_rad","wrist_elbow_root_frame_m","activity_ratio","generated_hold_seconds","source_hold_seconds"):
            group[k]=float(np.mean([r["source_metrics"][k] for r in rows]))
        for k in ("joint_jerk_mps3_p95","foot_skate_mps_p95"):group[k]=float(np.mean([r["audit"][k] for r in rows]))
        group["physical_nonregression_vs_T0"]=sum(all(r["audit"][k]<=baseline[(r["pair"],r["source"])]["audit"][k]+1e-6 for k in ("joint_jerk_mps3_p95","foot_skate_mps_p95")) for r in rows)
        groups[arm]=group
    traces=[]
    for item in p["traces"]:
        data=np.load(checked(item["trace"]));pred=data["predicted_root"].astype(float);final=data["guided_root"][-1].astype(float);s=data["source_root"].astype(float)
        delta=pred[0,0].T@final[0];yaw=lambda r:np.arctan2(r[0,2],r[2,2])
        valid=bool(np.linalg.norm(pred[0,0][(0,2),2])>.1 and np.linalg.norm(final[0][(0,2),2])>.1)
        traces.append(dict(pair=item["pair"],source=item["source"],arm=item["arm"],first_prediction_to_final_output_rad=float(Rotation.from_matrix(delta).magnitude()),
            yaw_defined=valid,yaw_change_rad=float(np.angle(np.exp(1j*(yaw(final[0])-yaw(pred[0,0]))))) if valid else None,
            final_relative_turn_rad=float(angles(final[:1].swapaxes(-1,-2)@final,s[:1].swapaxes(-1,-2)@s).mean()),causal_attribution=False))
    base={(b["pair"],b["source"]):b for b in p["bindings"] if b["arm"]=="T0"}
    factors=[foot_factors(base[(b["pair"],b["source"])],b) for b in p["bindings"] if b["arm"]!="T0"]
    write(out/"result.json",dict(groups=groups,traces=traces,foot_factors=factors,posthoc=True,used_for_checkpoint_selection=False,neural_generation_calls=0))
    videos=[]
    if render:
        import sys
        rp=Path(p["config"]["parent"]);prep=read(rp/"prepared/protocol.json");assets=read(checked(prep["assets"]));sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
        from motion.geometry import fk_24_np
        from dunhuang.research.primitive_reperformance.render import movie,audio_for
        from dunhuang.research.primitive_reperformance.diagnosis import canonical
        songs=read(checked(prep["music"]));song=next(s for s in songs if s["case"]==p["config"]["music"]);wave,sr=sf.read(checked(song["audio"]),dtype="float32",always_2d=True)
        for record,b in zip(result["records"],p["bindings"]):
            plan=read(checked(b["plan"]));event=plan["events"][0];full=np.load(checked(event["reference"]));a,z=event["source_interval"]
            reference,_,_,_=build(plan,read(rp/"single/protocol.json")["config"]["guide"])
            generated=np.load(checked(b["native"]));arrays=[canonical(fk_24_np(v),v) for v in (full[a:z],reference,generated)]
            name=str(record["pair"])+"_"+record["source"]+"_"+record["arm"];audio=out/(name+".wav");audio_for(wave,sr,0,max(map(len,arrays)),audio)
            file=out/(name+".mp4");frames=movie(arrays,audio,file,"R9 / "+name,wrist_glyphs=True)
            videos.append(dict(name=name,frames=frames,video=ref(file),native=record["native_ok"],physics=b["physical"],source_rad=record["source_metrics"]["mean_source_rad"]))
        cards=[]
        for v in videos:
            # Relative path resolves to the downloaded first-pass physical report, including saved T0.
            path=Path(v["physics"]["path"]);server="/home/disk/lsm/storage/DunhuangDanceGen/"
            rel=path.as_posix().removeprefix(server)
            import os
            physics=os.path.relpath(rel,out.relative_to(Path.cwd()))
            cards.append('<section><h2>'+v["name"]+' / Native='+str(v["native"])+' / source='+format(v["source_rad"],".3f")+' rad</h2><video controls preload="metadata" src="'+Path(v["video"]["path"]).name+'"></video><p><a href="'+physics+'">原Native报告</a></p></section>')
        (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>R9单事件来源对照</title><style>body{max-width:1200px;margin:24px auto;font:16px sans-serif;background:#f3f5f8}video{width:100%}section{background:white;padding:12px;margin:16px}</style><h1>R9完整单事件机制对照</h1><p>固定八个来源，T0/T1/T2/T2R/T3/T4全部展示，不按结果筛选。三栏为原始拟合ChangE、E1-W参考和实际生成。Root相对视图用于姿态对照，足滑以世界坐标Native为准；橙色虚线为已观测腕旋转的方向，手指未知。原始时钟与目标时钟不同，结束面板保持最后帧，仅为展示。T4保护的是局部下肢旋转，不代表世界足部不动。</p>'+''.join(cards),encoding="utf8")
    write(out/"protocol.json",dict(code_commit=commit(),main=ref(main/"source/protocol.json"),videos=videos,no_generation=True,source_truth="FITTED_ROTATIONS_ONLY",cultural_authenticity="UNKNOWN"))
    print("R9_POST_ANALYSIS",len(traces),len(videos),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--main",required=True);p.add_argument("--out",required=True);p.add_argument("--no-render",action="store_true");a=p.parse_args();run(a.main,a.out,not a.no_render)
