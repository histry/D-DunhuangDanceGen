"""R33 independent evidence, finite comparison media, and public statistics."""
import argparse,json,math,hashlib,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
from dunhuang.io import sha,write,checked
from dunhuang.data.sources import ref
from dunhuang.research.state_r27.common import read,fresh,commit,forbid_optimizers,same_arrays,readout_check
from release_r29 import video

def verify(root):
    root=Path(root);out=root/"independent_checks.json";assert not out.exists();forbid_optimizers()
    cfg=read("src/dunhuang/research/control_r33/config.json");bundles=0;checks=[]
    for stage in ("data","held_compilation_v2","evaluation"):
        a=read(root/stage/"result.json");b=read(root/(stage+"_replay")/"result.json");assert a==b
        for p in (root/stage).glob("*.npz"):same_arrays(p,root/(stage+"_replay")/p.name);bundles+=1
    c=read(root/"calibration/result.json");cr=read(root/"calibration_replay/result.json")
    assert c["training_admitted"] and c["optimizer_steps"]==8 and cr["optimizer_steps"]==0
    for k in ("scales","weight","finite_updates","initial_change_digest","training_admitted"):assert c[k]==cr[k]
    for x,y in zip(cr["actual_readouts"],c["observations"]):checks.append(readout_check(x,y))
    same_arrays(root/"calibration/arrays.npz",root/"calibration_replay/arrays.npz");bundles+=1
    assert max(cr["gradient_replay_relative_l2"].values())<=1e-5
    trained=[]
    for arm in ("Extra","Dyn"):
        p=root/f"models/{arm}";a=read(p/"result.json");b=read(root/f"replay/{arm}/result.json")
        assert a["optimizer_steps"]==600 and b["optimizer_steps"]==0
        for k in ("initial_digest","final_digest","frozen","curves","history","dynamic_weight","paired_source_counts"):assert a[k]==b[k]
        for name,h in a["files"].items():assert sha(p/name)==h
        for x,y in zip(b["actual_readouts"],a["observations"]):checks.append(readout_check(x,y))
        for v in p.glob("*.npz"):same_arrays(v,root/f"replay/{arm}"/v.name);bundles+=1
        assert not a["partial_Root_C4_truth"] and not a["context_trained"] and not a["general_reference_updated"]
        assert a["frozen"]==read(root/"data/result.json")["frozen"]
        trained.append(a)
    a,b=trained
    assert a["initial_digest"]==b["initial_digest"]==c["initial_change_digest"]
    ledger=lambda r:[{k:v for k,v in x.items() if k not in ("loss","parts")} for x in r["history"]]
    assert ledger(a)==ledger(b)
    counts=Counter(i for r in a["history"] if r["task"]=="paired" for i in r["cases"])
    assert len(counts)==16 and set(counts.values())=={75}
    assert a["dynamic_weight"]==0 and b["dynamic_weight"]==c["weight"]
    ev=read(root/"evaluation/result.json");assert ev["general_retention_exact"] and ev["NoRef_exact"] and ev["optimizer_steps"]==0
    assert not ev["held_used_for_training_or_selection"] and len({r["source_sha256"] for r in ev["held_records"]})==2
    for oldroot in (Path(cfg["r31"]),Path(cfg["r32"])):
        for name,record in read(oldroot/"artifacts_manifest.json")["files"].items():assert sha(oldroot/name)==record["sha256"]
    changed=subprocess.check_output(["git","diff",cfg["baseline"],"--name-only","--","src/dunhuang"],text=True).splitlines()
    assert all(p.startswith("src/dunhuang/research/control_r33/") for p in changed)
    components=[read(root/"data/result.json"),c,*trained,ev]
    assert sum(x["sampler_calls"] for x in components)==54 and sum(x["generated_trajectories"] for x in components)==264
    assert len(checks)==16 and bundles==10
    write(out,dict(code_commit=commit(),array_bundles_exact=bundles,update_pairs=16,update_checks=checks,
        generator_optimizer_steps=1200,calibration_optimizer_steps=8,replay_optimizer_steps=0,
        sampler_calls=54,generated_trajectories=264,replay_sampler_calls=54,replay_generated_trajectories=264,
        components=dict(preflight=[22,96],calibration=[0,0],training=[2,12],evaluation=[30,156]),
        paired_counts=dict(counts),gradient_replay_relative_l2=cr["gradient_replay_relative_l2"],
        fixed_general_and_NoRef_exact=True,parent_scientific_files_unchanged=True,
        single_backbone=True,context_trained=False,full_native_status="NOT_EVALUABLE_22_OF24",operational_qualified=False,whole_song_generations=0))
    print("R33_INDEPENDENT_CHECKS",bundles,len(checks),flush=True)

def render(root,out):
    root=Path(root);out=fresh(out);forbid_optimizers()
    from dunhuang.research.control_r33.common import population_data
    _,_,_,z=population_data();cfg=read("src/dunhuang/research/control_r33/config.json")
    arr=np.load(root/"evaluation/arrays.npz");held=np.load(root/"held_compilation_v2/arrays.npz")
    parents=np.load(checked(read("outputs/branch_r22_03/data/protocol.json")["parity"]))["parents"][:22].tolist()
    tracks={};audio={};notes={}
    for role in ("train","held"):
        for e in range(2):
            key=f"{role}_Source{e}_RootUNKNOWN";p=z["source22"][e,64:160] if role=="train" else held["local_points"][e]
            tracks[key]=p;notes[key]="SOURCE BODY | UNKNOWN ROOT | OUTSIDE CORE DISPLAY HOLDS"
    for role in ("train","held"):
        for m in (range(2) if role=="train" else range(1)):
            wav=Path(f"outputs/lodge_reference_r24_01/main/music{m}.wav")
            for e in range(2):
                for arm in ("Frozen","Extra","Dyn"):
                    key=f"{role}_m{m}_e{e}_{arm}";tracks[key]=arr[f"{role}_{arm}_correct_22"][3*m+e+1];audio[key]=wav
                    notes[key]="GENERATED | SAME MUSIC/NOISE | "+("ADAPTED SOURCES" if role=="train" else "R33 ADAPTATION HELD; HISTORICALLY EXPOSED CATALOG")
    for e in range(2):
        for mode in ("wrong_clock","swapped"):
            key=f"train_m0_e{e}_Dyn_{mode}";tracks[key]=arr[f"train_Dyn_{mode}_22"][e+1];audio[key]=Path("outputs/lodge_reference_r24_01/main/music0.wav");notes[key]="GENERATED COUNTERFACTUAL | "+mode
    for m in range(2):
        key=f"m{m}_FixedNoRef";tracks[key]=arr["train_Frozen_correct_22"][3*m];audio[key]=Path(f"outputs/lodge_reference_r24_01/main/music{m}.wav");notes[key]="FIXED R31 NOREF | EXACT FOR ALL ARMS"
    assert len(tracks)==28 and len(audio)==24
    c,s=math.cos(-.45),math.sin(-.45)
    def project(p):return np.stack((c*p[...,0]+s*p[...,2],p[...,1]-.18*(-s*p[...,0]+c*p[...,2])),-1)
    xy=np.concatenate([project(p).reshape(-1,2) for p in tracks.values()]);lo=xy.min(0);hi=xy.max(0);center=(lo+hi)/2
    scale=min(540/max(float(hi[0]-lo[0]),1e-6),440/max(float(hi[1]-lo[1]),1e-6));rows=[]
    for key,p in tracks.items():
        hold=len(p)==96
        if hold:p=np.concatenate((np.repeat(p[:1],64,0),p,np.repeat(p[-1:],96,0)))
        labels=[notes[key]+(" | Core64:160" if 64<=t<160 else (" | DISPLAY HOLD ONLY" if hold else " | Outside Core")) for t in range(256)]
        path=out/(key+".mp4");video(path,project(p),parents,"R33 | "+key,None,center,scale,audio.get(key),labels)
        rows.append(dict(key=key,file=path.name,video=ref(path),frames=256,has_audio=key in audio,full_decode_passed=True,display_holds_only=hold,notes=notes[key]))
        print("R33_VIDEO",key,flush=True)
    page="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>R33 来源保持与动态对照</title>
<style>body{font:16px system-ui;margin:24px;background:#f3f4f8;color:#163249}select,button{padding:8px;margin:5px}video{width:48%}p{line-height:1.7}</style>
<h1>R33：单一 Lodge 主干的来源保持与动态对照</h1><p>Frozen=R32固定ChangE参考；Extra=继续600步；Dyn=同600步加来源相对D1/D2监督。通用参考单独冻结路由，音乐、IO及主干固定。train来源参与适配；held仅为本轮适配录制隔离，来自历史曝光Catalog，不是新独立盲测。没有训练上下文支路或新时间映射网络。</p>
<p>原来源文件对应：train e0=female_FeiTian、e1=male_pipa_1；held e0=female_lotus、e1=female_meditation。名称来自资产身份，不等于已经审核的文化语义。</p><p>相同世界相机和比例，无平滑、逐帧重居中或贴地。Source只提供96帧身体姿态，Root未知，外侧仅显示定格。生成Root不是来源真值。两首音乐是已曝光非配对片段；音乐响应不等于协调性正确。22关节不能给出完整Native24合格结论，整曲0。右侧静音。</p>
<select id="left"></select><select id="right"></select><button id="play">同步播放／暂停</button><p id="detail"></p><video id="a" controls preload="metadata"></video><video id="b" controls muted preload="metadata"></video>
<script>const rows=PAYLOAD,l=document.querySelector('#left'),r=document.querySelector('#right'),a=document.querySelector('#a'),b=document.querySelector('#b');
for(const x of rows){l.add(new Option(x.key,x.file));r.add(new Option(x.key,x.file));}
l.value='train_m0_e1_Frozen.mp4';r.value='train_m0_e1_Dyn.mp4';
function load(){a.pause();b.pause();a.src=l.value;b.src=r.value;document.querySelector('#detail').textContent=l.value+' / '+r.value+' | Native24 NOT EVALUABLE';}
l.onchange=load;r.onchange=load;load();document.querySelector('#play').onclick=()=>{if(a.paused){b.currentTime=a.currentTime;Promise.all([a.play(),b.play()]).catch(()=>{});}else{a.pause();b.pause();}};
function tick(){if(!a.paused&&!b.paused&&Math.abs(a.currentTime-b.currentTime)>.08)b.currentTime=a.currentTime;requestAnimationFrame(tick);}tick();</script></html>"""
    (out/"index.html").write_text(page.replace("PAYLOAD",json.dumps(rows)),encoding="utf8")
    write(out/"protocol.json",dict(code_commit=commit(),records=rows,fixed_world_camera=True,motion_smoothing=False,generated_motion_ground_refit=False,live_browser_controls_verified=False))

def comparison(ev):
    rows=[]
    for role in ("train","held"):
        for arm in ("Frozen","Extra","Dyn"):
            for event in (0,1):
                records=[r for r in ev["records"] if r["role"]==role and r["arm"]==arm and r["mode"]=="correct" and r["event"]==event]
                def mean(path):
                    values=[]
                    for r in records:
                        v=r["errors"][event]
                        for key in path:v=v[key]
                        values.append(v)
                    return float(np.mean(values))
                row=dict(role=role,arm=arm,event=event,source_uid=records[0]["source_uid"],music_segments=len(records))
                for name,path in {"body_rad":["geometry","body22_rad"],"wrist_m":["geometry","world_wrist_rmse_m"],"activity_ratio":["geometry","body_activity_ratio"],"range_ratio":["geometry","arm_range_ratio"],"D1":["paired_body","1","rmse"],"D2":["paired_body","2","rmse"],"D3":["paired_body","3","rmse"],"high_frequency_error":["high_frequency_residual_power_ge6Hz"]}.items():row[name]=mean(path)
                for name in ("world","Root_translation","Root_relative_body"):row["Core_Jerk_"+name]=mean(["dynamics","regions","core_interior",name])
                row["full_Jerk"]=float(np.mean([r["quality"]["jerk_p95_mps3"] for r in records]))
                row["floor_speed_proxy"]=float(np.mean([r["floor"]["horizontal_speed_p95_mps"] for r in records if r["floor"]["horizontal_speed_p95_mps"] is not None])) if any(r["floor"]["horizontal_speed_p95_mps"] is not None for r in records) else None
                rows.append(row)
    return rows

def collect(root,out,receipts):
    root=Path(root);out=fresh(out)
    for name in receipts:
        p=Path(name);assert (p/"exit_status.txt").read_text().strip()=="0" and (p/"phase.txt").read_text().strip()=="completed"
    manifest=root/"artifacts_manifest.json";assert not manifest.exists()
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.rglob("*") if p.is_file()}
    write(manifest,dict(code_commit=commit(),files=files,file_count=len(files),total_bytes=sum(x["bytes"] for x in files.values())))
    models={}
    for arm in ("Extra","Dyn"):
        a=read(root/f"models/{arm}/result.json");models[arm]={k:v for k,v in a.items() if k!="history"}
        models[arm]["history_sha256"]=hashlib.sha256(json.dumps(a["history"],sort_keys=True).encode()).hexdigest()
        models[arm]["replay_checks"]=read(root/f"replay/{arm}/result.json")["replay_checks"]
    summary=dict(config=read("src/dunhuang/research/control_r33/config.json"),data=read(root/"data/result.json"),held=read(root/"held_compilation_v2/result.json"),
        calibration=read(root/"calibration/result.json"),models=models,evaluation=read(root/"evaluation/result.json"),checks=read(root/"independent_checks.json"),
        media=read(root/"videos/protocol.json"),comparison=comparison(read(root/"evaluation/result.json")),scientific_manifest=ref(manifest),human_reviewed=False,operational_qualified=False,whole_song_generations=0,full_native_status="NOT_EVALUABLE_22_OF24")
    write(out/"r33_summary.json",summary);write(out/"r33_checks.json",summary["checks"])
    failed=Path(str(root)+"_train_receipts")
    write(out/"r33_receipt.json",dict(collection_commit=commit(),summary_sha256=sha(out/"r33_summary.json"),scientific_manifest=ref(manifest),
        receipts={name:{p.name:sha(p) for p in Path(name).iterdir() if p.is_file()} for name in receipts},
        evidence={str(p.relative_to(root)):sha(p) for p in root.rglob("*.json")},
        launcher_logs={str(p):sha(p) for p in Path("outputs").glob("control_r33_*_launcher.log")},
        failed_train_launcher={p.name:sha(p) for p in failed.iterdir() if p.is_file()},
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False,whole_song_generations=0))
    print("R33_PUBLIC_STATS",sha(out/"r33_summary.json"),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",choices=["verify","render","collect"],required=True);p.add_argument("--root",required=True);p.add_argument("--out");p.add_argument("--receipts",nargs="*");a=p.parse_args()
    if a.task=="verify":verify(a.root)
    elif a.task=="render":render(a.root,a.out)
    else:collect(a.root,a.out,a.receipts)
