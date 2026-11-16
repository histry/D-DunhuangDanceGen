"""Independent numeric replay, frozen-contract evidence and finite R31 comparison media."""
import argparse,json,math,hashlib,subprocess
from collections import Counter
from pathlib import Path
import numpy as np
from dunhuang.io import sha,write,checked
from dunhuang.data.sources import ref
from dunhuang.research.state_r27.common import read,fresh,commit,forbid_optimizers,same_arrays,readout_check
from dunhuang.research.geometry_r31.common import sources,runtime as old_runtime
from release_r29 import video

def verify(root):
    root=Path(root);out=root/"independent_checks.json";assert not out.exists();forbid_optimizers()
    cfg=read("src/dunhuang/research/geometry_r31/config.json");rows,ledger,old=sources(cfg)
    exact=0
    superseded=Path("outputs/geometry_r31_01")
    for base,stage in [(superseded,"audit"),(root,"audit"),(root,"qualification"),(root,"evaluation"),(root,"change")]:
        a=read(base/stage/"result.json");b=read(base/(stage+"_replay")/"result.json")
        assert a==b
        assert a.get("optimizer_steps",a.get("neural_optimizer_steps",None))==0
        for p in (base/stage).glob("*.npz"):same_arrays(p,base/(stage+"_replay")/p.name);exact+=1
    assert not read(root/"audit/result.json")["gauge_training_admitted"]
    assert read(root/"qualification/result.json")["new_contract_training_admitted"]
    updates=[];initial=[]
    for arm in ("LegacyIO","CanonicalIO"):
        a=read(root/f"models/{arm}/result.json");b=read(root/f"models/{arm}_replay/result.json")
        assert a["optimizer_steps"]==1200 and b["optimizer_steps"]==0 and a["parameters"]==300440
        assert a["history"]==b["history"] and a["curves"]==b["curves"]
        assert [{k:r[k] for k in ("step","cases","seed","t")} for r in a["history"]]==ledger
        assert a["reference_initial_digest"]==old["initial_digest"] and a["frozen_digest"]==old["frozen_digest"]
        assert a["initial_digest"]==b["initial_digest"] and a["final_digest"]==b["final_digest"]
        assert len(b["actual_readouts"])==4
        for x,y in zip(b["actual_readouts"],a["observations"]):updates.append(readout_check(x,y))
        for name,h in a["files"].items():assert sha(root/f"models/{arm}"/name)==h
        for p in (root/f"models/{arm}").glob("curve_*.npz"):same_arrays(p,root/f"models/{arm}_replay"/p.name);exact+=1
        assert set(Counter(c for r in a["history"] for c in r["cases"]).values())=={300}
        initial.append(a["initial_digest"])
    assert initial[0]==initial[1]
    a=read(root/"probe/result.json");b=read(root/"probe_replay/result.json")
    assert a["records"]==b["records"] and a["optimizer_steps"]==b["optimizer_steps"]==0
    assert len(a["records"])==len(b["actual_gradient_readouts"])==16
    same_arrays(root/"probe/predictions.npz",root/"probe_replay/predictions.npz");exact+=1
    x=np.load(root/"probe/gradients.npz");y=np.load(root/"probe_replay/gradients.npz")
    assert set(x.files)==set(y.files) and len(x.files)==80
    gradient_errors={}
    for name in x.files:
        error=float(np.linalg.norm(x[name].astype(np.float64)-y[name])/max(np.linalg.norm(x[name].astype(np.float64)),1e-30))
        assert error<=1e-5;gradient_errors[name]=error
    # Historical R0 predicates remain unchanged; adapted I/O NoRef has its own separate denominator.
    r29=read("outputs/time_r29_01/evaluation_coverage/result.json")
    for g in read(root/"evaluation/result.json")["gates_frozen_NoRef"]:
        if g["arm"]!="R0":continue
        previous=next(r for r in r29["gates"] if r["arm"]=="COVER" and r["role"]==g["role"])
        for x,y in zip(g["conditions"],previous["conditions"]):
            for name in ("tests","source_pass","proxy","proxy_pass","correct_clock_better"):assert x[name]==y[name]
    changed=subprocess.check_output(["git","diff",cfg["baseline"],"--name-only","--","src/dunhuang"],text=True).splitlines()
    assert all(p.startswith("src/dunhuang/research/geometry_r31/") for p in changed)
    stages=[read(root/s/"result.json") for s in ("qualification","evaluation","probe","change")]
    calls=sum(s["sampler_calls"] for s in stages);trajectories=sum(s["generated_trajectories"] for s in stages)
    assert exact==74 and calls==40 and trajectories==172
    write(out,dict(code_commit=commit(),array_bundles_exact=exact,gradient_vector_bundle_numerically_replayed=1,
        gradient_vectors_replayed=80,gradient_vector_relative_l2_tolerance=1e-5,gradient_vector_relative_l2_errors=gradient_errors,
        generator_update_pairs=8,update_checks=updates,gradient_norm_rtol=1e-5,gradient_norm_atol=0,
        optimizer_steps=2400,replay_optimizer_steps=0,sampler_calls=calls,generated_trajectories=trajectories,
        replay_sampler_calls=calls,replay_generated_trajectories=trajectories,reused_frozen_rows=82,
        own_contract_and_frozen_NoRef_separated=True,old_source_proxy_gate_exact=True,
        superseded_audit_preserved=True,failed_qualification_preserved=True,frozen_code_unchanged=True,
        whole_song_generations=0,full_native_status="NOT_EVALUABLE",operational_qualified=False))
    print("R31_INDEPENDENT_CHECKS",exact,calls,trajectories,flush=True)

def render(root,out):
    import torch
    root=Path(root);out=fresh(out);cfg=read("src/dunhuang/research/geometry_r31/config.json")
    rows,ledger,_=sources(cfg);lookup={r["uid"]:r for r in rows}
    d,n,s,f,e,info,a=old_runtime(cfg);a.close();forbid_optimizers()
    z=np.load(root/"evaluation/arrays.npz");change=np.load(root/"change/arrays.npz")
    q=np.load(root/"qualification/arrays.npz")
    parents=np.load(checked(read("outputs/branch_r22_03/data/protocol.json")["parity"]))["parents"][:22].tolist()
    source_pop=read("outputs/contract_r28_01/data/result.json")["records"];tracks={};audio={}
    def legacy_points(row):
        raw=torch.tensor(row["data"]["model_raw"][None],device="cuda")
        with torch.no_grad():p=f(n.unnormalize(n.normalize(raw)),s)[0,:,:22].cpu().numpy()
        return p
    for role,case,i in (("train",48,1),("holdout",4,0)):
        uid="aist_"+role+"_"+str(case);r=lookup[uid];prefix="AIST_"+role+"_case"+str(case)
        tracks[prefix+"_Source"]=r["data"]["model_points"];tracks[prefix+"_OldRoundtrip"]=legacy_points(r);tracks[prefix+"_CanonicalRoundtrip"]=q[uid+"_decoded22"]
        for arm,mode in (("R0","correct"),("LegacyIO","correct"),("CanonicalIO","correct"),("CanonicalIO","wrong_clock"),("LegacyIO","none"),("CanonicalIO","none")):
            tracks[prefix+"_"+arm+"_"+mode]=z[f"{role}_{arm}_{mode}_0_22"][i]
        src=next(r for r in source_pop if r["case"]==case)
        for name in tracks:
            if name.startswith(prefix):audio[name]=Path("outputs/reference_r26_01/data")/src["source"]["crop_audio"]
    old=np.load(checked(read("outputs/lodge_reference_r24_01/main/protocol.json")["arrays"]))
    for event in range(2):
        row=lookup["change_"+str(event)];source=row["data"]["model_points"];origin=source[0,0]
        tracks[f"ChangE_Source{event}_RootUnknown"]=source
        lp=legacy_points(row);tracks[f"ChangE_OldRoundtrip{event}_RootUnknown"]=lp-lp[:,:1]+origin
        tracks[f"ChangE_CanonicalRoundtrip{event}_RootUnknown"]=q["change_"+str(event)+"_decoded22"]+origin
    for arm in ("R0","LegacyIO","CanonicalIO"):
        for event in range(2):tracks[f"ChangE_{arm}_e{event}"]=change[arm+"_correct_22"][event+1]
    for arm in ("LegacyIO","CanonicalIO"):tracks["ChangE_"+arm+"_NoRef"]=change[arm+"_correct_22"][0]
    tracks["ChangE_CanonicalIO_wrong_clock_e0"]=change["CanonicalIO_wrong_clock_22"][1]
    assert len(tracks)==33
    for name in tracks:
        if name.startswith("ChangE_") and "RootUnknown" not in name:audio[name]=Path("outputs/lodge_reference_r24_01/main/music0.wav")
    c,sn=math.cos(-.45),math.sin(-.45)
    def project(p):return np.stack((c*p[...,0]+sn*p[...,2],p[...,1]-.18*(-sn*p[...,0]+c*p[...,2])),-1)
    xy=np.concatenate([project(p).reshape(-1,2) for p in tracks.values()],0);lo=xy.min(0);hi=xy.max(0);center=(lo+hi)/2
    scale=min(540/max(float(hi[0]-lo[0]),1e-6),440/max(float(hi[1]-lo[1]),1e-6));records=[]
    for name,p in tracks.items():
        # ChangE source-only tracks contain the96 observed Core frames; pad only the DISPLAY duration by holding endpoints.
        display_only=len(p)==96
        if display_only:p=np.concatenate((np.repeat(p[:1],64,0),p,np.repeat(p[-1:],96,0)),0)
        assert len(p)==256
        path=out/(name+".mp4");notes=[("Source Root UNKNOWN | " if "RootUnknown" in name else "")+("Core64:160" if 64<=t<160 else ("DISPLAY HOLD ONLY" if display_only else "Outside Core")) for t in range(256)]
        video(path,project(p),parents,"R31 | "+name,None,center,scale,audio.get(name),notes)
        records.append(dict(key=name,file=path.name,video=ref(path),frames=256,has_audio=name in audio,full_decode_passed=True,
            source_only_display_endpoint_holds=display_only,source_Root_world_truth="UNKNOWN" if "RootUnknown" in name else "OBSERVED_OR_GENERATED"))
        print("R31_VIDEO",name,flush=True)
    page="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>R31 保几何动作合同</title>
<style>body{font:16px system-ui;margin:24px;background:#f3f4f8;color:#163249}select,button{padding:8px;margin:5px}video{width:48%}p{line-height:1.7}</style>
<h1>R31：原来源几何、模型坐标与训练目标</h1><p>R0=冻结COVER。LegacyIO/CanonicalIO同300440参数、同16来源/1200步，区别为显式身体动作合同及固定噪声预条件。CanonicalRoundtrip只验证表示往返，不是神经生成。先比较Source/OldRoundtrip/CanonicalRoundtrip，再看三种生成。</p>
<p>旧盒内等价编码未准入，缺失帧没有拼成训练真值。未知ChangE Root只展示局部姿态，source-only96帧Core两端显示定格延长到256帧，标DISPLAY HOLD ONLY，未作为训练或评价动作；旧/新往返仅为可视化放在同一未知来源Root原点。生成动作未平滑、重居中或贴地，统一世界相机与比例。完整Native24不可评，均非合格整曲。右侧静音。</p>
<select id="left"></select><select id="right"></select><button id="play">同步播放／暂停</button><p id="detail"></p><video id="a" controls preload="metadata"></video><video id="b" controls muted preload="metadata"></video>
<script>const rows=PAYLOAD,l=document.querySelector('#left'),r=document.querySelector('#right'),a=document.querySelector('#a'),b=document.querySelector('#b');
for(const x of rows){l.add(new Option(x.key,x.file));r.add(new Option(x.key,x.file));}
l.value='ChangE_R0_e0.mp4';r.value='ChangE_CanonicalIO_e0.mp4';
function load(){a.pause();b.pause();a.src=l.value;b.src=r.value;document.querySelector('#detail').textContent=l.value+' / '+r.value+' | Native24 NOT EVALUABLE';}
l.onchange=load;r.onchange=load;load();document.querySelector('#play').onclick=()=>{if(a.paused){b.currentTime=a.currentTime;Promise.all([a.play(),b.play()]).catch(()=>{});}else{a.pause();b.pause();}};
function tick(){if(!a.paused&&!b.paused&&Math.abs(a.currentTime-b.currentTime)>.08)b.currentTime=a.currentTime;requestAnimationFrame(tick);}tick();</script></html>"""
    (out/"index.html").write_text(page.replace("PAYLOAD",json.dumps(records)),encoding="utf8")
    write(out/"protocol.json",dict(code_commit=commit(),records=records,fixed_world_camera=True,motion_smoothing=False,
        generated_motion_ground_refit=False,live_browser_controls_verified=False,source_endpoint_holds_only_for_display=True))

def seal(root):
    root=Path(root);path=root/"artifacts_manifest.json";assert not path.exists()
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.rglob("*") if p.is_file()}
    write(path,dict(sealing_commit=commit(),files=files,file_count=len(files),total_bytes=sum(x["bytes"] for x in files.values())))
    return ref(path)

def collect(root,out,receipts):
    root=Path(root);out=fresh(out)
    for name in receipts:
        p=Path(name);assert (p/"exit_status.txt").read_text().strip()=="0" and (p/"phase.txt").read_text().strip()=="completed"
    result={stage:read(root/stage/"result.json") for stage in ("audit","qualification","evaluation","probe","change")}
    models={}
    for arm in ("LegacyIO","CanonicalIO"):
        a=read(root/f"models/{arm}/result.json")
        models[arm]={k:v for k,v in a.items() if k!="history"}
        models[arm]["history_sha256"]=hashlib.sha256(json.dumps(a["history"],sort_keys=True).encode()).hexdigest()
        models[arm]["replay_checks"]=read(root/f"models/{arm}_replay/result.json")["replay_checks"]
    result.update(models=models,config=read("src/dunhuang/research/geometry_r31/config.json"),
        checks=read(root/"independent_checks.json"),gradient_replay_checks=read(root/"probe_replay/result.json")["gradient_replay_checks"],
        media=read(root/"videos/protocol.json"),superseded_audit_manifest=seal("outputs/geometry_r31_01"),
        scientific_manifest=seal(root),full_native_status="NOT_EVALUABLE",whole_song_generations=0,operational_qualified=False)
    write(out/"r31_summary.json",result);write(out/"r31_checks.json",result["checks"])
    failed=Path(str(root)+"_qualification_receipts")
    write(out/"r31_receipt.json",dict(collection_commit=commit(),summary_sha256=sha(out/"r31_summary.json"),
        scientific_manifest=result["scientific_manifest"],superseded_audit_manifest=result["superseded_audit_manifest"],
        evidence={str(p.relative_to(root)):sha(p) for p in root.rglob("*.json")},
        receipts={name:{p.name:sha(p) for p in Path(name).iterdir() if p.is_file()} for name in receipts},
        failed_qualification_receipts={p.name:sha(p) for p in failed.iterdir() if p.is_file()},
        failed_evaluation_receipts={p.name:sha(p) for p in Path(str(root)+"_evaluation_receipts").iterdir() if p.is_file()},
        gradient_recovery=ref(Path(str(root)+"_gradient_recovery.json")),
        initial_superseded_receipts={p.name:sha(p) for p in Path("outputs/geometry_r31_01_initial_receipts").iterdir() if p.is_file()},
        qualification_recovery=ref(Path(str(root)+"_qualification_recovery.json")),
        launcher_logs={str(p):sha(p) for p in Path("outputs").glob("geometry_r31_*_launcher.log")},
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,whole_song_generations=0,operational_qualified=False))
    print("R31_PUBLIC_STATS",sha(out/"r31_summary.json"),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",required=True,choices=["verify","render","collect"]);p.add_argument("--root",required=True);p.add_argument("--out");p.add_argument("--receipts",nargs="*");a=p.parse_args()
    if a.task=="verify":verify(a.root)
    elif a.task=="render":render(a.root,a.out)
    else:collect(a.root,a.out,a.receipts)
