"""Read-only independent verification, shared-camera media and public scientific receipts."""
import argparse,json,math,subprocess,hashlib
from pathlib import Path
import numpy as np
from dunhuang.io import sha,write,checked
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.state_r27.common import readout_check,metrics
from render_r23 import video
from dunhuang.research.contract_r28.evidence import factorial_dir

def verify(root,out=None):
    root=Path(root);out=Path(out) if out else root/"independent_checks.json";assert not out.exists();forbid_optimizers();bundles=0
    recovery=read(root/"recovery_inputs.json") if (root/"recovery_inputs.json").exists() else dict(sealed_files={},failed_sampler_calls=0,failed_generated_trajectories=0)
    for p,h in recovery["sealed_files"].items():assert sha(p)==h
    for stage in ("data",factorial_dir(root).name,"readout","fine_audio","evaluation","change"):
        a=read(root/stage/"result.json");b=read(root/(stage+"_replay")/"result.json");assert a==b
        assert b["optimizer_steps"]==0
        for p in list((root/stage).rglob("*.npz"))+list((root/stage).rglob("*.npy")):
            other=root/(stage+"_replay")/p.relative_to(root/stage)
            if p.suffix==".npz":same_arrays(p,other)
            else:assert np.array_equal(np.load(p),np.load(other))
            bundles+=1
    checks=[];models=[]
    for arm in ("T0","T1"):
        a=read(root/f"models/{arm}/result.json");b=read(root/f"models/{arm}_replay/result.json")
        assert a["optimizer_steps"]==1200 and b["optimizer_steps"]==0
        assert a["initial_digest"]==b["initial_digest"] and a["final_digest"]==b["final_digest"]
        assert a["history"]==b["history"] and a["curves"]==b["curves"]
        assert len(b["actual_readouts"])==len(a["observations"])==4
        for name,h in a["files"].items():assert sha(root/f"models/{arm}"/name)==h
        for x,y in zip(b["actual_readouts"],a["observations"]):checks.append(readout_check(x,y))
        for p in (root/f"models/{arm}").glob("curve_*.npz"):same_arrays(p,root/f"models/{arm}_replay"/p.name);bundles+=1
        assert all(x["update_norm"]>0 for x in a["observations"])
        models.append(a)
    assert models[0]["initial_digest"]==models[1]["initial_digest"] and models[0]["frozen_digest"]==models[1]["frozen_digest"]
    assert [(x["step"],x["seed"],x["t"],x["cases"]) for x in models[0]["history"]]==[(x["step"],x["seed"],x["t"],x["cases"]) for x in models[1]["history"]]
    audit=read(root/"input_domain_audit.json");replay=read(root/"input_domain_audit_replay.json")
    assert {k:v for k,v in audit.items() if k!="code_commit"}=={k:v for k,v in replay.items() if k!="code_commit"}
    paths=["src/dunhuang/research/reference_r26","src/dunhuang/research/state_r27","src/dunhuang/research/lodge_adapter_r25","src/dunhuang/research/branch_r22","src/dunhuang/research/local_r23","src/dunhuang/research/lodge_reference_r24","src/dunhuang/edge_backend.py","src/dunhuang/planning","src/dunhuang/sampling.py"]
    assert not subprocess.check_output(["git","diff","a870ed2c0888049923fa42c0f8056a4162ed4e0b","--"]+paths,text=True)
    # New random seeds alone are not a physical novelty certificate; compare every new reference to both old same-music references.
    cfg=read(root/"data/protocol.json")["config"]
    for key in ("r26","r27"):
        parent=Path(cfg[key]);assert sha(parent/"artifacts_manifest.json")==cfg[key+"_manifest_sha256"]
        manifest=read(parent/"artifacts_manifest.json")["files"]
        for name in (("models/C1/adapter.pt","samples/arrays.npz") if key=="r26" else ("models/P/adapter.pt","teachers/teacher.npz")):
            assert sha(parent/name)==manifest[name]["sha256"]
    old=np.load(Path(cfg["r27"])/"teachers/teacher.npz");fac=np.load(factorial_dir(root)/"arrays.npz")
    novelty=[]
    for i in range(4):
        angles=[metrics(fac["new_raw139"][i],fac["new_points22"][i],old["raw139"][j],old["points22"][j])["body22_rad"] for j in range(2*(i//2),2*(i//2)+2)]
        novelty.append(dict(teacher=i,nearest_seen_same_music_body_rad=min(angles),distinct_from_seen=min(angles)>.05))
    assert all(x["distinct_from_seen"] for x in novelty)
    gauge=[]
    for pop in ("seen","new_reference"):
        x=fac[f"{pop}_P_raw_correct_points"][:,64:160];y=fac[f"{pop}_P_canonical_correct_points"][:,64:160]
        gauge.append(dict(population=pop,final_root_relative_FK_difference_rms_m=float(np.sqrt(np.mean(((x-x[:,:,:1])-(y-y[:,:,:1]))**2)))))
    stages=[read(root/s/"result.json") for s in (factorial_dir(root).name,"evaluation","change")]
    write(out,dict(code_commit=commit(),array_bundles_exact=bundles,new_reference_physical_novelty=novelty,canonical_reference_final_response_diagnostic=gauge,actual_update_pairs_verified=8,training_optimizer_steps=2400,replay_optimizer_steps=0,
        matched_training_initialization_and_step_noise_clock=True,Core_C4_supervision=False,
        gradient_checks=checks,gradient_norm_rtol=1e-5,gradient_norm_atol=0,non_gradient_readouts_exact=True,
        successful_sampler_calls=sum(x["sampler_calls"] for x in stages),successful_generated_trajectories=sum(x["generated_trajectories"] for x in stages),
        replay_sampler_calls=sum(x["sampler_calls"] for x in stages),replay_generated_trajectories=sum(x["generated_trajectories"] for x in stages),
        failed_sampler_calls=recovery["failed_sampler_calls"],failed_generated_trajectories=recovery["failed_generated_trajectories"],failed_optimizer_steps=0,
        historical_formal_and_R27_paths_unchanged=True,source_holdout_scope="ADAPTATION_ONLY_EXPOSED_TO_R26_NOT_INDEPENDENT_BLIND",
        full_native_status="NOT_EVALUABLE",whole_song_generations=0,operational_qualified=False))
    print("R28_INDEPENDENT_CHECKS",bundles,flush=True)

def render(root,out):
    root=Path(root);out=fresh(out);cfg=read(root/"data/protocol.json")["config"]
    fac=np.load(factorial_dir(root)/"arrays.npz");ev=np.load(root/"evaluation/arrays.npz");ch=np.load(root/"change/arrays.npz")
    old=np.load(checked(read(Path(cfg["r24"])/"main/protocol.json")["arrays"]))
    seen=np.load(Path(cfg["r27"])/"teachers/teacher.npz")
    parent=np.load(checked(read("outputs/branch_r22_03/data/protocol.json")["parity"]))["parents"][:22].tolist()
    tracks={};audios={}
    teacher_audio=checked(read(Path(cfg["r27"])/"teachers/result.json")["sources"][0]["audio"])
    for pop,label,source in (("seen","SeenTeacher",seen["points22"][0]),("new_reference","NewTeacher",fac["new_points22"][0])):
        tracks[label+"_Source"]=source;tracks[label+"_NoRef"]=fac[pop+"_baseline_points"][0]
        for rep in ("raw","canonical"):tracks[label+"_P_"+rep]=fac[f"{pop}_P_{rep}_correct_points"][0]
        for name in list(tracks):
            if name.startswith(label):audios[name]=teacher_audio
    population=read(root/"data/result.json")["records"]
    for role,case,label in (("train",92,"AIST_Train92"),("holdout_exposed_R26",4,"AIST_Holdout4")):
        row=next(r for r in population if r["case"]==case);data=np.load(root/"data"/row["arrays"])
        tracks[label+"_Source"]=data["model_points"];tracks[label+"_NoRef"]=ev[f"{role}_baseline_none_0_22"][0]
        for arm in ("R26_C1","T0","T1"):tracks[label+"_"+arm]=ev[f"{role}_{arm}_correct_0_22"][0]
        audio=Path(cfg["r26"])/"data"/row["source"]["crop_audio"]
        for name in list(tracks):
            if name.startswith(label):audios[name]=audio
    for i in range(2):tracks[f"ChangE_Source{i}_RootUnknown"]=old["source22"][i]
    tracks["ChangE_NoRef"]=ch["T1_correct_22"][0]
    audio=Path(cfg["r24"])/"main/music0.wav"
    for arm in ("R26_C1","P","T0","T1"):
        for e in range(2):
            name=f"ChangE_{arm}_e{e}";tracks[name]=ch[arm+"_correct_22"][e+1];audios[name]=audio
    audios["ChangE_NoRef"]=audio;assert len(tracks)==29
    c,s=math.cos(-.45),math.sin(-.45)
    def project(x):return np.stack((c*x[...,0]+s*x[...,2],x[...,1]-.18*(-s*x[...,0]+c*x[...,2])),-1)
    xy=np.stack([project(v) for v in tracks.values()]);lo=xy.min((0,1,2));hi=xy.max((0,1,2));center=(lo+hi)/2
    scale=min(540/max(float(hi[0]-lo[0]),1e-6),440/max(float(hi[1]-lo[1]),1e-6));records=[]
    for key,points in tracks.items():
        p=out/(key+".mp4")
        label="New reference, never trained" if key.startswith("NewTeacher") else "Adaptation held-out, R26 exposed" if key.startswith("AIST_Holdout") else "ChangE Root UNKNOWN" if "RootUnknown" in key else "Fixed research comparison"
        notes=[label+" | "+("Core64:160" if 64<=t<160 else "Outside Core") for t in range(256)]
        video(p,project(points),parent,"R28 | "+key,None,center,scale,audios.get(key),notes)
        records.append(dict(key=key,file=p.name,video=ref(p),frames=256,full_decode_passed=True,has_audio=key in audios));print("R28_VIDEO",key,flush=True)
    page="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>R28 来源泛化与输入合同</title>
<style>body{font:16px system-ui;background:#f3f4f8;color:#163249;margin:24px}select,button{padding:8px;margin:5px}video{width:48%}p{line-height:1.7}</style>
<h1>R28：来源泛化、参考规范化与真实小集学习</h1><p>SeenTeacher是R27训练过的参考，NewTeacher是相同音乐下未用于适配的新教师动作。raw/canonical只改参考表示、不改物理旋转。T0/T1从同一C1出发，各四条真实AIST、1200步，分别旧输入包与公开模型floor0/C4输入包。</p>
<p>AIST Holdout仅相对本轮适配持出，历史R26已曝光。ChangE Source仅有旋转，Root=0展示且UNKNOWN，不能以其足部高度认定足滑；生成视频为固定模型坐标。统一世界相机/比例，无平滑、无逐帧居中、无生成动作贴地。完整Native24不可评价，未跑整曲或取得文化风格资格。</p>
<select id="left"></select><select id="right"></select><button id="play">同步播放／暂停</button><p id="detail"></p><video id="a" controls preload="metadata"></video><video id="b" controls muted preload="metadata"></video>
<script>const rows=PAYLOAD,l=document.querySelector('#left'),r=document.querySelector('#right'),a=document.querySelector('#a'),b=document.querySelector('#b');
for(const x of rows){l.add(new Option(x.key,x.file));r.add(new Option(x.key,x.file));}
l.value='AIST_Train92_T0.mp4';r.value='AIST_Train92_T1.mp4';
function load(){a.pause();b.pause();a.src=l.value;b.src=r.value;document.querySelector('#detail').textContent=l.value+' / '+r.value+' | Native24 NOT EVALUABLE';}
l.onchange=load;r.onchange=load;load();document.querySelector('#play').onclick=()=>{if(a.paused){b.currentTime=a.currentTime;Promise.all([a.play(),b.play()]).catch(()=>{});}else{a.pause();b.pause();}};
function tick(){if(!a.paused&&!b.paused&&Math.abs(a.currentTime-b.currentTime)>.08)b.currentTime=a.currentTime;requestAnimationFrame(tick);}tick();</script></html>"""
    (out/"index.html").write_text(page.replace("PAYLOAD",json.dumps(records)),encoding="utf8")
    write(out/"protocol.json",dict(code_commit=commit(),records=records,fixed_world_camera=True,motion_smoothing=False,generated_motion_ground_refit=False,
        ChangE_source_Root_unobserved=True,browser_interaction_tested=False))

def collect(root,out,receipts):
    root=Path(root);out=fresh(out)
    for path in receipts:
        p=Path(path);assert (p/"exit_status.txt").read_text().strip()=="0" and (p/"phase.txt").read_text().strip()=="completed"
    models={}
    for arm in ("T0","T1"):
        a=read(root/f"models/{arm}/result.json");b=read(root/f"models/{arm}_replay/result.json")
        models[arm]={k:v for k,v in a.items() if k!="history"}
        models[arm]["history_sha256"]=hashlib.sha256(json.dumps(a["history"],sort_keys=True).encode()).hexdigest()
        models[arm]["original_result"]=ref(root/f"models/{arm}/result.json");models[arm]["replay_checks"]=b["replay_checks"]
    evaluation_arrays=np.load(root/"evaluation/arrays.npz");cfg=read(root/"data/protocol.json")["config"];axis_rows=[]
    pop=read(root/"data/result.json")["records"]
    for row in read(root/"evaluation/result.json")["records"]:
        case=row["case"];source=next(r for r in pop if r["case"]==case)
        truth=np.load(root/"data"/source["arrays"])["model_raw"]
        stem=f'{row["role"]}_{row["arm"]}_{row["mode"]}_{row["seed_index"]}_139'
        i=(cfg["train_cases"] if row["role"]=="train" else cfg["holdout_cases"]).index(case)
        delta=evaluation_arrays[stem][i,64:160,4:7]-truth[64:160,4:7]
        axes=np.sqrt(np.mean(delta**2,axis=0))
        assert np.isclose(float(np.sqrt(np.mean(axes**2))),row["root_rmse_m"],rtol=1e-6,atol=1e-7)
        axis_rows.append(dict(case=case,role=row["role"],arm=row["arm"],mode=row["mode"],seed_index=row["seed_index"],root_xyz_rmse_m=axes.tolist(),scope="DESCRIPTIVE_FIXED_COMMON_MODEL0_SOURCE_REFERENCE_NOT_NEW_QUALIFICATION"))
    result=dict(root_axis_readouts=axis_rows,data=read(root/"data/result.json"),factorial=read(factorial_dir(root)/"result.json"),P_readout=read(root/"readout/result.json"),
        preclip_audit=read(root/"input_domain_audit.json"),fine_audio=read(root/"fine_audio/result.json"),models=models,
        training_admission=read(root/"training_admission.json"),population_admission=read(root/"population_admission.json"),
        evaluation=read(root/"evaluation/result.json"),change=read(root/"change/result.json"),
        checks=read(root/("independent_checks_final.json" if (root/"independent_checks_final.json").exists() else "independent_checks.json")),media=read(root/"videos/protocol.json"),recovery=read(root/"recovery_inputs.json") if (root/"recovery_inputs.json").exists() else None,
        operational_qualified=False,full_native_status="NOT_EVALUABLE",whole_song_generations=0)
    manifest=root/"artifacts_manifest.json";assert not manifest.exists()
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.rglob("*") if p.is_file()}
    write(manifest,dict(code_commit=commit(),files=files,file_count=len(files),total_bytes=sum(x["bytes"] for x in files.values())))
    result["scientific_manifest"]=ref(manifest);write(out/"r28_summary.json",result);write(out/"r28_checks.json",result["checks"])
    write(out/"r28_receipt.json",dict(collection_commit=commit(),summary_sha256=sha(out/"r28_summary.json"),scientific_manifest=ref(manifest),
        evidence={str(p.relative_to(root)):sha(p) for p in root.rglob("*.json")},
        receipts={p:{x.name:sha(x) for x in Path(p).iterdir() if x.is_file()} for p in receipts},
        failed_receipts={x.name:sha(x) for x in Path(str(root)+"_initial_receipts").iterdir() if x.is_file()},
        launcher_logs={str(p):sha(p) for p in Path("outputs").glob(root.name+"*_launcher.log")},
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,whole_song_generations=0,operational_qualified=False))
    print("R28_PUBLIC_STATS",sha(out/"r28_summary.json"),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",required=True,choices=["verify","render","collect"]);p.add_argument("--root",required=True);p.add_argument("--out");p.add_argument("--receipts",nargs="*");a=p.parse_args()
    if a.task=="verify":verify(a.root,a.out)
    elif a.task=="render":render(a.root,a.out)
    else:collect(a.root,a.out,a.receipts)
