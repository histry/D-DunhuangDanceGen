"""Independent frozen replay verification, finite shared-camera media and public metadata."""
import argparse,json,math,subprocess,hashlib
from pathlib import Path
import numpy as np
from dunhuang.io import sha,write,checked
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.state_r27.common import readout_check
from render_r23 import video

def verify(root):
    root=Path(root);out=root/"independent_checks.json";assert not out.exists();forbid_optimizers();bundles=0
    for stage in ("diagnostic","teachers","positive","transfer"):
        a=read(root/stage/"result.json");b=read(root/(stage+"_replay")/"result.json");assert a==b and b["optimizer_steps"]==0
        for p in (root/stage).glob("*.npz"):same_arrays(p,root/(stage+"_replay")/p.name);bundles+=1
    models=read(root/"release_admission.json")["models"];assert models==["P"]
    a=read(root/"models/P/result.json");b=read(root/"models/P_replay/result.json")
    assert a["optimizer_steps"]==1200 and b["optimizer_steps"]==0
    assert a["initial_digest"]==b["initial_digest"] and a["final_digest"]==b["final_digest"]
    assert a["curves"]==b["curves"] and a["history"]==b["history"] and a["frozen_digest"]==b["frozen_digest"]
    for name,h in a["files"].items():assert sha(root/"models/P"/name)==h
    clean=read(root/"pruned_replay_receipt.json");assert clean["observed_process_exit_status"]==0 and clean["optimizer_steps"]==0
    differences=[]
    for actual,expected in zip(b["actual_readouts"],a["observations"]):differences.append(readout_check(actual,expected))
    assert len(differences)==4 and all(x["update_norm"]>0 for x in a["observations"])
    for p in (root/"models/P").glob("curve_*.npz"):same_arrays(p,root/"models/P_replay"/p.name);bundles+=1
    with np.load(root/"teachers/teacher.npz") as t,np.load(root/"teachers/teacher_trace.npz") as z:
        assert np.array_equal(t["target"],z["final"]) and t["target"].shape==(4,256,139)
    assert read(root/"teachers/result.json")["teacher_eval_seed_disjoint"]
    diag=read(root/"diagnostic/result.json");assert diag["uninstrumented_sampler_exact"]
    assert len(diag["matched_readouts"])==256 and all(x["geometry_gradient_finite"] for x in diag["matched_readouts"])
    frozen=["src/dunhuang/research/reference_r26","src/dunhuang/research/lodge_adapter_r25","src/dunhuang/research/branch_r22","src/dunhuang/research/local_r23","src/dunhuang/research/lodge_reference_r24","src/dunhuang/edge_backend.py","src/dunhuang/planning","src/dunhuang/sampling.py"]
    assert not subprocess.check_output(["git","diff","dabfd297de027a7b50ffe8ef28c461a4491425f4","--"]+frozen,text=True)
    stages={s:read(root/s/"result.json") for s in ("diagnostic","teachers","positive","transfer")}
    write(out,dict(code_commit=commit(),array_bundles_exact=bundles,matched_diagnostic_readouts=256,actual_update_pairs_verified=4,
        gradient_norm_rtol=1e-5,gradient_norm_atol=0,gradient_checks=differences,non_gradient_readouts_exact=True,
        training_optimizer_steps=1200,replay_optimizer_steps=0,unexecuted_capacity_continuation_unroll_steps=0,
        sampler_calls=sum(x["sampler_calls"] for x in stages.values()),replay_sampler_calls=sum(x["sampler_calls"] for x in stages.values()),
        generated_trajectories=sum(x["generated_trajectories"] for x in stages.values()),replay_generated_trajectories=sum(x["generated_trajectories"] for x in stages.values()),
        reused_transfer_trajectories=stages["transfer"]["reused_trajectories"],coupled_teacher_target_exact=True,teacher_eval_seeds_disjoint=True,
        historical_and_formal_paths_unchanged=True,full_native_status="NOT_EVALUABLE",whole_song_generations=0,operational_qualified=False))
    print("R27_INDEPENDENT_CHECKS",bundles,flush=True)

def render(root,out):
    root=Path(root);out=fresh(out);models=read(root/"release_admission.json")["models"]
    teacher=np.load(root/"teachers/teacher.npz");positive=np.load(root/"positive/arrays.npz");transfer=np.load(root/"transfer/arrays.npz")
    cfg=read(root/"teachers/protocol.json")["config"];old=np.load(checked(read(Path(cfg["r24"])/"main/protocol.json")["arrays"]))
    parent=np.load(checked(read("outputs/branch_r22_03/data/protocol.json")["parity"]))["parents"][:22].tolist()
    tracks={};audios={};scope={}
    for i in range(2):
        key=f"Teacher_source{i}";tracks[key]=teacher["points22"][i];scope[key]="MODEL_TEACHER_NOT_REAL_CONTACT"
    audio=checked(read(root/"teachers/result.json")["sources"][0]["audio"])
    tracks["Teacher_none"]=positive["baseline_none_0_points"][0];audios["Teacher_none"]=audio
    for arm in ["R26_C1"]+models:
        for i in range(2):
            key=f"Teacher_{arm}_e{i}";tracks[key]=positive[f"{arm}_correct_0_points"][i];audios[key]=audio
    for i in range(2):tracks[f"ChangE_source{i}"]=old["source22"][i]
    audio=Path(cfg["r24"])/"main/music0.wav"
    tracks["ChangE_none"]=transfer["change_R26_C0_correct_22"][0];audios["ChangE_none"]=audio
    for arm in ["R26_C1"]+models:
        for ev in range(2):
            key=f"ChangE_{arm}_e{ev}";tracks[key]=transfer[f"change_{arm}_correct_22"][ev+1];audios[key]=audio
    assert len(tracks)==14
    c,s=math.cos(-.45),math.sin(-.45)
    def project(x):return np.stack((c*x[...,0]+s*x[...,2],x[...,1]-.18*(-s*x[...,0]+c*x[...,2])),-1)
    xy=np.stack([project(v) for v in tracks.values()]);lo=xy.min((0,1,2));hi=xy.max((0,1,2));center=(lo+hi)/2
    scale=min(540/max(float(hi[0]-lo[0]),1e-6),440/max(float(hi[1]-lo[1]),1e-6));records=[]
    for key,points in tracks.items():
        p=out/(key+".mp4")
        notes=[("MODEL TEACHER - not measured contact" if key.startswith("Teacher") else "ChangE retimed proxy - exposed music")+" | "+("Core" if 64<=t<160 else "outside Core") for t in range(256)]
        video(p,project(points),parent,"R27 | "+key,None,center,scale,audios.get(key),notes)
        records.append(dict(key=key,file=p.name,video=ref(p),frames=256,full_decode_passed=True,has_audio=key in audios))
        print("R27_VIDEO",key,flush=True)
    page="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>R27 完整输入状态与参考学习</title>
<style>body{font:16px system-ui;background:#f3f4f8;color:#163249;margin:24px}select,button{padding:8px;margin:5px}video{width:48%}p{line-height:1.7}</style>
<h1>R27：同分布参考学习与真实来源迁移</h1><p>Teacher为冻结Lodge在真实音乐下生成的完整139D模型目标，C4不是实测接触。P仅在两首Train音乐、四条教师轨迹上训练1200步；本页教师采样使用不同噪声。ChangE为固定历史来源迁移，不用于选模。</p>
<p>22关节骨架，Native24仍不可评，未取得敦煌文化风格或整曲资格。Source的Core外持姿不是生成目标。统一世界相机与比例，无平滑、无逐帧居中。右侧静音；建议比较相同前缀与事件编号。</p>
<select id="left"></select><select id="right"></select><button id="play">同步播放／暂停</button><p id="detail"></p><video id="a" controls preload="metadata"></video><video id="b" controls muted preload="metadata"></video>
<script>const rows=PAYLOAD,l=document.querySelector('#left'),r=document.querySelector('#right'),a=document.querySelector('#a'),b=document.querySelector('#b');
for(const x of rows){l.add(new Option(x.key,x.file));r.add(new Option(x.key,x.file));}
l.value='ChangE_R26_C1_e0.mp4';r.value='ChangE_P_e0.mp4';
function load(){a.pause();b.pause();a.src=l.value;b.src=r.value;document.querySelector('#detail').textContent=l.value+' / '+r.value+' | Native24 NOT EVALUABLE';}
l.onchange=load;r.onchange=load;load();document.querySelector('#play').onclick=()=>{if(a.paused){b.currentTime=a.currentTime;Promise.all([a.play(),b.play()]).catch(()=>{});}else{a.pause();b.pause();}};
function tick(){if(!a.paused&&!b.paused&&Math.abs(a.currentTime-b.currentTime)>.08)b.currentTime=a.currentTime;requestAnimationFrame(tick);}tick();</script></html>"""
    (out/"index.html").write_text(page.replace("PAYLOAD",json.dumps(records)),encoding="utf8")
    write(out/"protocol.json",dict(code_commit=commit(),records=records,motion_smoothing=False,fixed_world_camera=True,browser_interaction_tested=False))

def collect(root,out,receipts):
    root=Path(root);out=fresh(out)
    for folder in receipts:
        p=Path(folder);assert (p/"exit_status.txt").read_text().strip()=="0" and (p/"phase.txt").read_text().strip()=="completed"
    a=read(root/"models/P/result.json");b=read(root/"models/P_replay/result.json")
    model={k:v for k,v in a.items() if k not in ("history",)}
    model["history_sha256"]=hashlib.sha256(json.dumps(a["history"],sort_keys=True).encode()).hexdigest()
    model["original_result"]=ref(root/"models/P/result.json");model["replay_checks"]=b["replay_checks"]
    result=dict(diagnostic=read(root/"diagnostic/result.json"),teachers=read(root/"teachers/result.json"),model=model,
        positive=read(root/"positive/result.json"),decision=read(root/"decision.json"),admission=read(root/"release_admission.json"),
        transfer=read(root/"transfer/result.json"),checks=read(root/"independent_checks.json"),media=read(root/"videos/protocol.json"),
        human_reviewed=False,operational_qualified=False,whole_song_generations=0,full_native_status="NOT_EVALUABLE")
    assert read(root/"pruned_replay_receipt.json")["optimizer_steps"]==0
    manifest=root/"artifacts_manifest.json";assert not manifest.exists()
    files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in root.rglob("*") if p.is_file()}
    write(manifest,dict(code_commit=commit(),files=files,file_count=len(files),total_bytes=sum(p["bytes"] for p in files.values()),temporary_ssh_files=0))
    result["scientific_manifest"]=ref(manifest);result["pruned_replay"]=read(root/"pruned_replay_receipt.json")
    write(out/"r27_summary.json",result);write(out/"r27_checks.json",result["checks"])
    write(out/"r27_receipt.json",dict(collection_commit=commit(),summary_sha256=sha(out/"r27_summary.json"),
        evidence={str(p.relative_to(root)):sha(p) for p in root.rglob("*.json")},
        receipts={folder:{p.name:sha(p) for p in Path(folder).iterdir() if p.is_file()} for folder in receipts},
        launcher_logs={str(p):sha(p) for folder in receipts for p in [Path(folder.removesuffix("_receipts")+"_launcher.log")] if p.exists()},
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,training_optimizer_steps=1200,replay_optimizer_steps=0,operational_qualified=False,whole_song_generations=0))
    print("R27_PUBLIC_STATS",sha(out/"r27_summary.json"),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",required=True,choices=["verify","render","collect"]);p.add_argument("--root",required=True);p.add_argument("--out");p.add_argument("--receipts",nargs="*");a=p.parse_args()
    if a.task=="verify":verify(a.root)
    elif a.task=="render":render(a.root,a.out)
    else:collect(a.root,a.out,a.receipts)
