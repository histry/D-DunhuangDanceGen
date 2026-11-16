"""Finite R30 evidence, deterministic derived support and immutable comparison media."""
import argparse,json,math,hashlib,subprocess
from collections import Counter
from pathlib import Path
import numpy as np
from dunhuang.io import sha,write
from dunhuang.data.sources import ref
from dunhuang.research.state_r27.common import read,fresh,commit,forbid_optimizers,same_arrays,readout_check
from dunhuang.research.continuity_r30.common import values,parent
from release_r29 import video

def verify(root):
    root=Path(root);out=root/"independent_checks.json";assert not out.exists();forbid_optimizers()
    cfg=read("src/dunhuang/research/continuity_r30/config.json");previous=parent(cfg)
    model_names=["R1"]+(["R2"] if (root/"models/R2/result.json").exists() else [])
    bundles=0;checks=[];stages=["audit","reading_train","reading_holdout","temporal_evaluation","support","probe","change"]
    for stage in stages:
        a=read(root/stage/"result.json");b=read(root/(stage+"_replay")/"result.json")
        assert a==b and a["optimizer_steps"]==0
        for p in (root/stage).glob("*.npz"):same_arrays(p,root/(stage+"_replay")/p.name);bundles+=1
    a=read(root/"calibration/result.json");b=read(root/"calibration_replay/result.json")
    assert a["weights"]==b["weights"] and a["chosen_baseline"]==b["chosen_baseline"] and a["optimizer_steps"]==b["optimizer_steps"]==0
    for x,y in zip(b["actual_readouts"],a["readouts"]):checks.append(readout_check(x,y))
    assert len(b["actual_readouts"])==4
    same_arrays(root/"calibration/arrays.npz",root/"calibration_replay/arrays.npz");bundles+=1
    old=read(previous/"models/COVER/result.json");ledger=lambda x:[{k:r[k] for k in ("step","cases","seed","t")} for r in x["history"]]
    for arm in model_names:
        a=read(root/f"models/{arm}/result.json");b=read(root/f"models/{arm}_replay/result.json")
        assert a["optimizer_steps"]==1200 and b["optimizer_steps"]==0 and a["parameters"]==265328
        assert a["history"]==b["history"] and a["curves"]==b["curves"] and ledger(a)==ledger(old)
        assert a["initial_digest"]==b["initial_digest"]==old["initial_digest"]
        assert a["frozen_digest"]==old["frozen_digest"] and a["final_digest"]==b["final_digest"]
        for name,h in a["files"].items():assert sha(root/f"models/{arm}"/name)==h
        assert len(b["actual_readouts"])==4
        for x,y in zip(b["actual_readouts"],a["observations"]):checks.append(readout_check(x,y))
        for p in (root/f"models/{arm}").glob("curve_*.npz"):same_arrays(p,root/f"models/{arm}_replay"/p.name);bundles+=1
        assert set(Counter(c for r in a["history"] for c in r["cases"]).values())=={300}
        assert set(Counter((c,r["t"]) for r in a["history"] for c in r["cases"]).values())=={6}
    # The exact historical five source tests and four physical proxy tests are retained.
    original=read(previous/"evaluation_coverage/result.json")
    for folder in ("reading_train","reading_holdout"):
        current=read(root/folder/"result.json")
        for g in current["gates"]:
            if g["arm"]!="R0":continue
            oldgate=next(t for t in original["gates"] if t["arm"]=="COVER" and t["role"]==g["role"])
            for x,y in zip(g["conditions"],oldgate["conditions"]):
                for k in ("tests","source_pass","proxy","proxy_pass","correct_clock_better"):assert x[k]==y[k]
    changed=subprocess.check_output(["git","diff",cfg["baseline"],"--name-only","--","src/dunhuang"],text=True).splitlines()
    assert all(p.startswith("src/dunhuang/research/continuity_r30/") for p in changed)
    stage_results=[read(root/s/"result.json") for s in stages]
    calls=sum(s["sampler_calls"] for s in stage_results);trajectories=sum(s["generated_trajectories"] for s in stage_results)
    assert bundles==(16 if "R2" in model_names else 12)
    assert calls==(45 if "R2" in model_names else 25) and trajectories==(192 if "R2" in model_names else 106)
    write(out,dict(code_commit=commit(),array_bundles_exact=bundles,optimizer_steps=1200*len(model_names),replay_optimizer_steps=0,
        generator_update_pairs=4*len(model_names),calibration_gradient_readouts=4,gradient_checks=checks,
        gradient_norm_rtol=1e-5,gradient_norm_atol=0,non_gradient_readouts_exact=True,
        sampler_calls=calls,generated_trajectories=trajectories,replay_sampler_calls=calls,replay_generated_trajectories=trajectories,
        reused_frozen_rows=sum(x.get("reused_trajectories",0) for x in stage_results),original_source_and_proxy_gate_exact=True,
        immutable_R29_manifest_sha256=cfg["r29_manifest_sha256"],frozen_code_unchanged=True,
        whole_song_generations=0,full_native_status="NOT_EVALUABLE",operational_qualified=False))
    print("R30_INDEPENDENT_CHECKS",bundles,calls,trajectories,flush=True)

def render(root,out):
    root=Path(root);out=fresh(out);cfg=read("src/dunhuang/research/continuity_r30/config.json")
    oldcfg=read("src/dunhuang/research/time_r29/config.json")
    oldpop=read(Path(oldcfg["r28"])/"data/result.json")["records"]
    from dunhuang.io import checked
    parents=np.load(checked(read("outputs/branch_r22_03/data/protocol.json")["parity"]))["parents"][:22].tolist()
    temporal=np.load(root/"temporal_evaluation/arrays.npz");change=np.load(root/"change/arrays.npz")
    tracks={};audio={};hasR2=(root/"models/R2/result.json").exists()
    for role,case in (("train",92),("holdout",4)):
        data,_=values(cfg,"aist",role);prefix="AIST_"+role;tracks[prefix+"_Source"]=data[0]["model_points"]
        z=np.load(root/("reading_train" if role=="train" else "reading_holdout")/"arrays.npz")
        for arm,mode in (("NoRef","none"),("R0","correct"),("R1","correct"),("R1","wrong_clock")):tracks[prefix+"_"+arm+"_"+mode]=z[f"{role}_{arm}_{mode}_0_22"][0]
        tracks[prefix+"_RootDelta0_correct"]=temporal[f"{role}_RootDelta0_correct_0_22"][0]
        if hasR2:
            for mode in ("correct","wrong_clock"):tracks[prefix+"_R2_"+mode]=temporal[f"{role}_R2_{mode}_0_22"][0]
        row=next(r for r in oldpop if r["case"]==case)
        for name in tracks:
            if name.startswith(prefix):audio[name]=Path(oldcfg["r26"])/"data"/row["source"]["crop_audio"]
    from dunhuang.io import checked
    old=np.load(checked(read(Path(oldcfg["r24"])/"main/protocol.json")["arrays"]))
    for i in range(2):tracks[f"ChangE_Source{i}_RootUnknown"]=old["source22"][i]
    tracks["ChangE_NoRef"]=change["R0_correct_22"][0]
    for arm in (["R0","R1","R2"] if hasR2 else ["R0","R1"]):
        for event in range(2):tracks[f"ChangE_{arm}_e{event}"]=change[arm+"_correct_22"][event+1]
    for arm in (["R1","R2"] if hasR2 else ["R1"]):tracks[f"ChangE_{arm}_wrong_clock_e0"]=change[arm+"_wrong_clock_22"][1]
    assert len(tracks)==(27 if hasR2 else 20)
    for name in tracks:
        if name.startswith("ChangE_") and "Source" not in name:audio[name]=Path(oldcfg["r24"])/"main/music0.wav"
    c,s=math.cos(-.45),math.sin(-.45)
    def project(p):return np.stack((c*p[...,0]+s*p[...,2],p[...,1]-.18*(-s*p[...,0]+c*p[...,2])),-1)
    xy=np.stack([project(p) for p in tracks.values()]);lo=xy.min((0,1,2));hi=xy.max((0,1,2));center=(lo+hi)/2
    scale=min(540/max(float(hi[0]-lo[0]),1e-6),440/max(float(hi[1]-lo[1]),1e-6));records=[]
    for name,points in tracks.items():
        path=out/(name+".mp4");notes=[("Source Root UNKNOWN | " if "RootUnknown" in name else "")+("Core64:160" if 64<=i<160 else "Outside Core") for i in range(256)]
        video(path,project(points),parents,"R30 | "+name,None,center,scale,audio.get(name),notes)
        records.append(dict(key=name,file=path.name,video=ref(path),frames=256,has_audio=name in audio,full_decode_passed=True));print("R30_VIDEO",name,flush=True)
    page="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>R30 连续参考与来源动力学</title>
<style>body{font:16px system-ui;margin:24px;background:#f3f4f8;color:#163249}select,button{padding:8px;margin:5px}video{width:48%}p{line-height:1.7}</style>
<h1>R30：连续读取参考与来源动力学保持</h1><p>R0为冻结R29 COVER。R1相同起点/16来源/1200步，只改Core局部插值＋全局门控。R2若存在，为Train封存目标权重的同预算配对Root/全身时序对照。RootDelta0只抑制当前状态/CFG分支的直接Root参考增量，C4/body保持；并非独立NoRef Root轨迹。</p>
<p>AIST为来源机制实验；ChangE为固定两历史原语迁移。完整Native24不可评，不是合格整曲。Source Root UNKNOWN只展示局部来源。统一世界相机/比例，没有平滑、生成后贴地、逐帧居中；右侧静音。先看来源动作，再比较R0/R1/R2。</p>
<select id="left"></select><select id="right"></select><button id="play">同步播放／暂停</button><p id="detail"></p><video id="a" controls preload="metadata"></video><video id="b" controls muted preload="metadata"></video>
<script>const rows=PAYLOAD,l=document.querySelector('#left'),r=document.querySelector('#right'),a=document.querySelector('#a'),b=document.querySelector('#b');
for(const x of rows){l.add(new Option(x.key,x.file));r.add(new Option(x.key,x.file));}
l.value='ChangE_R0_e0.mp4';r.value=rows.some(x=>x.key==='ChangE_R2_e0')?'ChangE_R2_e0.mp4':'ChangE_R1_e0.mp4';
function load(){a.pause();b.pause();a.src=l.value;b.src=r.value;document.querySelector('#detail').textContent=l.value+' / '+r.value+' | Native24 NOT EVALUABLE';}
l.onchange=load;r.onchange=load;load();document.querySelector('#play').onclick=()=>{if(a.paused){b.currentTime=a.currentTime;Promise.all([a.play(),b.play()]).catch(()=>{});}else{a.pause();b.pause();}};
function tick(){if(!a.paused&&!b.paused&&Math.abs(a.currentTime-b.currentTime)>.08)b.currentTime=a.currentTime;requestAnimationFrame(tick);}tick();</script></html>"""
    (out/"index.html").write_text(page.replace("PAYLOAD",json.dumps(records)),encoding="utf8")
    write(out/"protocol.json",dict(code_commit=commit(),records=records,fixed_world_camera=True,motion_smoothing=False,generated_motion_ground_refit=False,live_browser_controls_verified=False))

def collect(root,out,receipts):
    root=Path(root);out=fresh(out)
    for name in receipts:
        p=Path(name);assert (p/"exit_status.txt").read_text().strip()=="0" and (p/"phase.txt").read_text().strip()=="completed"
    stages=["audit","reading_train","calibration","reading_holdout","temporal_evaluation","probe","change"]
    result={stage:read(root/stage/"result.json") for stage in stages};models={}
    for arm in ("R1","R2"):
        if not (root/f"models/{arm}/result.json").exists():continue
        a=read(root/f"models/{arm}/result.json")
        models[arm]={k:v for k,v in a.items() if k!="history"}
        models[arm]["history_sha256"]=hashlib.sha256(json.dumps(a["history"],sort_keys=True).encode()).hexdigest()
        models[arm]["replay_checks"]=read(root/f"models/{arm}_replay/result.json")["replay_checks"]
    result.update(config=read("src/dunhuang/research/continuity_r30/config.json"),training_population=read(parent(read("src/dunhuang/research/continuity_r30/config.json"))/"coverage_data/result.json"),models=models,calibration_replay_checks=read(root/"calibration_replay/result.json")["replay_checks"],
        source_support=read(root/"support/result.json"),checks=read(root/"independent_checks.json"),media=read(root/"videos/protocol.json"),
        full_native_status="NOT_EVALUABLE",operational_qualified=False,whole_song_generations=0)
    manifest=root/"artifacts_manifest.json";assert not manifest.exists()
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.rglob("*") if p.is_file()}
    write(manifest,dict(code_commit=commit(),files=files,file_count=len(files),total_bytes=sum(x["bytes"] for x in files.values())))
    result["scientific_manifest"]=ref(manifest);write(out/"r30_summary.json",result);write(out/"r30_checks.json",result["checks"])
    write(out/"r30_receipt.json",dict(collection_commit=commit(),summary_sha256=sha(out/"r30_summary.json"),scientific_manifest=ref(manifest),
        evidence={str(p.relative_to(root)):sha(p) for p in root.rglob("*.json")},
        receipts={name:{p.name:sha(p) for p in Path(name).iterdir() if p.is_file()} for name in receipts},
        launcher_logs={str(p):sha(p) for p in Path("outputs").glob(root.name+"*_launcher.log")},
        receipt_directory_relocation=ref(Path(str(root)+"_receipt_relocation.json")),
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,whole_song_generations=0,operational_qualified=False))
    print("R30_PUBLIC_STATS",sha(out/"r30_summary.json"),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",required=True,choices=["verify","render","collect"]);p.add_argument("--root",required=True);p.add_argument("--out");p.add_argument("--receipts",nargs="*");a=p.parse_args()
    if a.task=="verify":verify(a.root)
    elif a.task=="render":render(a.root,a.out)
    else:collect(a.root,a.out,a.receipts)
