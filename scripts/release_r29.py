"""Independent replay, finite media and public provenance; frozen baselines stay intact."""
import argparse,json,math,hashlib,subprocess
from collections import Counter
from pathlib import Path
import numpy as np
from dunhuang.io import sha,write,checked
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,fresh,commit,forbid_optimizers,same_arrays
from dunhuang.research.state_r27.common import readout_check
from PIL import Image,ImageDraw

def video(path,points,parents,label,ok,center,scale,audio,annotations):
    """R29 shared world camera; captions retain actual dataset identity."""
    assert ok is None and not path.exists()
    cmd=["ffmpeg","-nostdin","-threads","2","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","600x600","-r","30","-i","-"]
    if audio:cmd += ["-i",str(audio),"-map","0:v:0","-map","1:a:0"]
    cmd += ["-c:v","libx264","-pix_fmt","yuv420p","-preset","veryfast","-crf","20","-threads","2"]
    if audio:cmd += ["-c:a","aac","-shortest"]
    proc=subprocess.Popen(cmd+[str(path)],stdin=subprocess.PIPE)
    try:
        for t,frame in enumerate(points):
            im=Image.new("RGB",(600,600),"#f2f4f7");draw=ImageDraw.Draw(im)
            draw.text((12,12),label,fill="#173249")
            draw.text((12,34),"Native24 NOT EVALUABLE | 22-joint research window",fill="#70402e")
            draw.text((12,56),annotations[t],fill="#70402e")
            pts=[(round(300+(v[0]-center[0])*scale),round(310-(v[1]-center[1])*scale)) for v in frame]
            for j in range(1,len(frame)):draw.line([pts[parents[j]],pts[j]],fill="#285e99" if j%2 else "#9c5133",width=4)
            for x,y in pts:draw.ellipse((x-3,y-3,x+3,y+3),fill="#22394c")
            draw.text((12,553),f"{t+1}/{len(points)} | fixed world camera, common scale",fill="#415b6b")
            draw.text((12,576),"No smoothing / no recentering / unqualified research",fill="#415b6b")
            proc.stdin.write(np.asarray(im).tobytes())
    finally:proc.stdin.close()
    assert proc.wait()==0
    count=subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","csv=p=0",str(path)],text=True,timeout=60)
    assert int(count.strip())==256
    subprocess.run(["ffmpeg","-nostdin","-threads","2","-v","error","-i",str(path),"-f","null","-"],check=True,timeout=60)

def verify(root):
    root=Path(root);out=root/"independent_checks.json";assert not out.exists();forbid_optimizers();bundles=0;checks=[]
    covered=read(root/"coverage_admission.json")["coverage_admitted"]
    stages0=["fine_data","parity","evaluation_aist","evaluation_fine","probe","change"]+(["coverage_data","evaluation_coverage","change_coverage"] if covered else [])
    for stage in stages0:
        a=read(root/stage/"result.json");b=read(root/(stage+"_replay")/"result.json");assert a==b and a["optimizer_steps"]==0
        for p in (root/stage).glob("*.npz"):same_arrays(p,root/(stage+"_replay")/p.name);bundles+=1
    r=read(root/"readout/result.json");rr=read(root/"readout_replay/result.json")
    assert r["decoder_optimizer_steps"]==1800 and rr["decoder_optimizer_steps"]==0 and rr["generator_optimizer_steps"]==0
    for name,a in r["models"].items():
        b=rr["models"][name];assert a["initial_digest"]==b["initial_digest"] and a["final_digest"]==b["final_digest"] and a["curves"]==b["curves"]
        for fn,h in a["files"].items():assert sha(root/"readout"/name/fn)==h
        for x,y in zip(b["actual_readouts"],a["observations"]):checks.append(readout_check(x,y))
    same_arrays(root/"readout/arrays.npz",root/"readout_replay/arrays.npz");bundles+=1
    trained=[]
    for name in (["B1","F1"]+(["COVER"] if covered else [])):
        a=read(root/f"models/{name}/result.json");b=read(root/f"models/{name}_replay/result.json")
        assert a["optimizer_steps"]==1200 and b["optimizer_steps"]==0 and a["parameters"]==265328
        assert a["initial_digest"]==b["initial_digest"] and a["final_digest"]==b["final_digest"]
        assert a["curves"]==b["curves"] and a["history"]==b["history"]
        assert len(a["observations"])==len(b["actual_readouts"])==4
        for fn,h in a["files"].items():assert sha(root/f"models/{name}"/fn)==h
        for x,y in zip(b["actual_readouts"],a["observations"]):checks.append(readout_check(x,y))
        for p in (root/f"models/{name}").glob("curve_*.npz"):same_arrays(p,root/f"models/{name}_replay"/p.name);bundles+=1
        trained.append(a)
    cfg=read(root/"fine_data/protocol.json")["config"];old=read(Path(cfg["r28"])/"models/T1/result.json")
    assert sha(Path(cfg["r28"])/"artifacts_manifest.json")==cfg["r28_manifest_sha256"]
    assert all(m["initial_digest"]==old["initial_digest"] for m in trained)
    assert all(m["frozen_digest"]==old["frozen_digest"] for m in trained)
    ledger=lambda x:[(r["step"],r["seed"],r["t"],r["cases"]) for r in x["history"]]
    assert ledger(trained[0])==ledger(old)
    if covered:
        cover=trained[-1]
        assert cover["recordings"]==16 and cover["music_groups"]==8 and cover["per_target_presentations"]==300
        assert set(Counter(c for row in cover["history"] for c in row["cases"]).values())=={300}
        assert set(Counter((c,row["t"]) for row in cover["history"] for c in row["cases"]).values())=={6}
        assert len({c for row in cover["history"] for c in row["cases"]})==16
        data=read(root/"coverage_data/result.json")
        assert len({r["recording_uid"] for r in data["records"]})==16 and len({r["source_group"] for r in data["records"]})==8
        assert not set(data["plan"]["groups"])&set(data["plan"]["excluded_groups"])
    assert len(checks)==(21 if covered else 17)
    assert bundles==(43 if covered else 22)
    paths=["src/dunhuang/research/contract_r28","src/dunhuang/research/state_r27","src/dunhuang/research/reference_r26","src/dunhuang/research/lodge_adapter_r25","src/dunhuang/research/branch_r22","src/dunhuang/edge_backend.py","src/dunhuang/planning","src/dunhuang/sampling.py"]
    assert not subprocess.check_output(["git","diff",cfg["baseline"],"--"]+paths,text=True)
    stages=[read(root/s/"result.json") for s in (["parity","evaluation_aist","evaluation_fine","probe","change"]+(["evaluation_coverage","change_coverage"] if covered else []))]
    assert sum(x["sampler_calls"] for x in stages)==(76 if covered else 57)
    assert sum(x["generated_trajectories"] for x in stages)==(322 if covered else 240)
    write(out,dict(code_commit=commit(),array_bundles_exact=bundles,decoder_optimizer_steps=1800,generator_optimizer_steps=1200*len(trained),replay_optimizer_steps=0,
        actual_decoder_update_pairs_verified=9,actual_generator_update_pairs_verified=4*len(trained),gradient_checks=checks,gradient_norm_rtol=1e-5,gradient_norm_atol=0,non_gradient_readouts_exact=True,
        successful_sampler_calls=sum(x["sampler_calls"] for x in stages),successful_generated_trajectories=sum(x["generated_trajectories"] for x in stages),
        replay_sampler_calls=sum(x["sampler_calls"] for x in stages),replay_generated_trajectories=sum(x["generated_trajectories"] for x in stages),
        B0_exact_original_R28=True,unique_reused_frozen_trajectories=60,frozen_row_reuses=120 if covered else 60,B1_original_R28_clock_noise_ledger_exact=True,
        actual_reverse_state_predictions_exact=True,frozen_baseline_and_formal_code_unchanged=True,whole_song_generations=0,full_native_status="NOT_EVALUABLE",operational_qualified=False))
    print("R29_INDEPENDENT_CHECKS",bundles,flush=True)

def render(root,out):
    root=Path(root);out=fresh(out);cfg=read(root/"fine_data/protocol.json")["config"]
    x=np.load(root/"evaluation_aist/arrays.npz");y=np.load(root/"evaluation_fine/arrays.npz");z=np.load(root/"change/arrays.npz")
    old=np.load(checked(read(Path(cfg["r24"])/"main/protocol.json")["arrays"]))
    parent=np.load(checked(read("outputs/branch_r22_03/data/protocol.json")["parity"]))["parents"][:22].tolist()
    population=read(Path(cfg["r28"])/"data/result.json")["records"]
    covered=read(root/"coverage_admission.json")["coverage_admitted"]
    tracks={};audios={}
    for role,case in (("train",92),("holdout",4)):
        row=next(r for r in population if r["case"]==case);data=np.load(Path(cfg["r28"])/"data"/row["arrays"])
        label="AIST_"+role;tracks[label+"_Source"]=data["model_points"]
        for arm,mode in (("baseline","none"),("B0","correct"),("B1","correct"),("B1","wrong_clock"),("B1","swapped")):tracks[label+"_"+arm+"_"+mode]=x[f"{role}_{arm}_{mode}_0_22"][0]
        if covered:
            cov=np.load(root/"evaluation_coverage/arrays.npz")
            for mode in ("correct","wrong_clock"):tracks[label+"_COVER_"+mode]=cov[f"{role}_COVER_{mode}_0_22"][0]
        for key in list(tracks):
            if key.startswith(label):audios[key]=Path(cfg["r26"])/"data"/row["source"]["crop_audio"]
    import soundfile as sf
    finepop=read(root/"fine_data/result.json")["records"]
    audio_sources=Path("inputs/contract_r28/finedance/music_wav")
    for role,seq in (("train","001"),("holdout","003")):
        row=next(r for r in finepop if r["role"]==role and r["recording"]==seq and r["start_frame"]==0)
        data=np.load(root/"fine_data"/row["arrays"]);label="FineDance_"+role;tracks[label+"_Source"]=data["model_points"]
        for arm,mode in (("baseline","none"),("F0","correct"),("F1","correct"),("F1","wrong_clock"),("F1","swapped")):tracks[label+"_"+arm+"_"+mode]=y[f"{role}_{arm}_{mode}_0_22"][0]
        wave,sr=sf.read(audio_sources/(seq+".wav"),dtype="float32",always_2d=True);wav=root/(label+"_viewer_audio.wav");assert not wav.exists();sf.write(wav,wave[:round(256/30*sr)],sr,subtype="FLOAT")
        for key in list(tracks):
            if key.startswith(label):audios[key]=wav
    for ev in range(2):tracks[f"ChangE_Source{ev}_RootUnknown"]=old["source22"][ev]
    tracks["ChangE_NoRef"]=z["B1_correct_22"][0]
    for arm in ("B0","B1","F1"):
        for ev in range(2):tracks[f"ChangE_{arm}_e{ev}"]=z[arm+"_correct_22"][ev+1]
    for arm in ("B1","F1"):tracks[f"ChangE_{arm}_wrong_clock_e0"]=z[arm+"_wrong_clock_22"][1]
    if covered:
        cov=np.load(root/"change_coverage/arrays.npz")
        for ev in range(2):tracks[f"ChangE_COVER_e{ev}"]=cov["COVER_correct_22"][ev+1]
        tracks["ChangE_COVER_wrong_clock_e0"]=cov["COVER_wrong_clock_22"][1]
    for key in list(tracks):
        if key.startswith("ChangE_") and "Source" not in key:audios[key]=Path(cfg["r24"])/"main/music0.wav"
    assert len(tracks)==(42 if covered else 35)
    c,s=math.cos(-.45),math.sin(-.45)
    def project(p):return np.stack((c*p[...,0]+s*p[...,2],p[...,1]-.18*(-s*p[...,0]+c*p[...,2])),-1)
    xy=np.stack([project(p) for p in tracks.values()]);lo=xy.min((0,1,2));hi=xy.max((0,1,2));center=(lo+hi)/2
    scale=min(540/max(float(hi[0]-lo[0]),1e-6),440/max(float(hi[1]-lo[1]),1e-6));records=[]
    for key,p in tracks.items():
        path=out/(key+".mp4");notes=[("Source Root UNKNOWN | " if "RootUnknown" in key else "")+("Core64:160" if 64<=t<160 else "Outside Core") for t in range(256)]
        video(path,project(p),parent,"R29 | "+key,None,center,scale,audios.get(key),notes)
        records.append(dict(key=key,file=path.name,video=ref(path),frames=256,has_audio=key in audios,full_decode_passed=True));print("R29_VIDEO",key,flush=True)
    page="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>R29 显式时间参考控制</title>
<style>body{font:16px system-ui;background:#f3f4f8;color:#163249;margin:24px}select,button{padding:8px;margin:5px}video{width:48%}p{line-height:1.7}</style>
<h1>R29：读对参考时间，保持来源动作</h1><p>B0为封存R28 T1，B1为相同数据/起点/1200步，只增加Core时间偏置。F0为冻结C1初始化，F1在FineDance两Train录制四窗口训练；另两录制持出，预训练曝光未知。correct、wrong_clock和swapped均为完整采样。</p>
<p>单个256帧研究窗口，完整Native24不可评；不等于整曲或文化风格通过。ChangE Source Root=0仅展示且UNKNOWN。COVER若存在，为已登记16录制/8音乐组的同总预算覆盖研究，不是按持出择优。相同世界相机与比例，无平滑、逐帧居中或生成动作贴地。右侧静音。</p>
<select id="left"></select><select id="right"></select><button id="play">同步播放／暂停</button><p id="detail"></p><video id="a" controls preload="metadata"></video><video id="b" controls muted preload="metadata"></video>
<script>const rows=PAYLOAD,l=document.querySelector('#left'),r=document.querySelector('#right'),a=document.querySelector('#a'),b=document.querySelector('#b');
for(const x of rows){l.add(new Option(x.key,x.file));r.add(new Option(x.key,x.file));}
l.value='AIST_train_B0_correct.mp4';r.value='AIST_train_B1_correct.mp4';
function load(){a.pause();b.pause();a.src=l.value;b.src=r.value;document.querySelector('#detail').textContent=l.value+' / '+r.value+' | Native24 NOT EVALUABLE';}
l.onchange=load;r.onchange=load;load();document.querySelector('#play').onclick=()=>{if(a.paused){b.currentTime=a.currentTime;Promise.all([a.play(),b.play()]).catch(()=>{});}else{a.pause();b.pause();}};
function tick(){if(!a.paused&&!b.paused&&Math.abs(a.currentTime-b.currentTime)>.08)b.currentTime=a.currentTime;requestAnimationFrame(tick);}tick();</script></html>"""
    (out/"index.html").write_text(page.replace("PAYLOAD",json.dumps(records)),encoding="utf8")
    write(out/"protocol.json",dict(code_commit=commit(),records=records,fixed_world_camera=True,generated_motion_ground_refit=False,motion_smoothing=False,live_browser_controls_verified=False))

def collect(root,out,receipts):
    root=Path(root);out=fresh(out)
    for folder in receipts:
        p=Path(folder);assert (p/"exit_status.txt").read_text().strip()=="0" and (p/"phase.txt").read_text().strip()=="completed"
    covered=read(root/"coverage_admission.json")["coverage_admitted"]
    models={}
    for arm in (["B1","F1"]+(["COVER"] if covered else [])):
        a=read(root/f"models/{arm}/result.json");models[arm]={k:v for k,v in a.items() if k!="history"}
        models[arm]["history_sha256"]=hashlib.sha256(json.dumps(a["history"],sort_keys=True).encode()).hexdigest()
        models[arm]["original_result"]=ref(root/f"models/{arm}/result.json");models[arm]["replay_checks"]=read(root/f"models/{arm}_replay/result.json")["replay_checks"]
    result={s:read(root/s/"result.json") for s in (["fine_data","parity","readout","evaluation_aist","evaluation_fine","probe","change"]+(["coverage_data","evaluation_coverage","change_coverage"] if covered else []))}
    result.update(models=models,readout_replay_checks=read(root/"readout_replay/result.json")["replay_checks"],coverage=read(root/"coverage_admission.json"),
        checks=read(root/"independent_checks.json"),media=read(root/"videos/protocol.json"),full_native_status="NOT_EVALUABLE",whole_song_generations=0,operational_qualified=False)
    manifest=root/"artifacts_manifest.json";assert not manifest.exists()
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.rglob("*") if p.is_file()}
    write(manifest,dict(code_commit=commit(),files=files,file_count=len(files),total_bytes=sum(x["bytes"] for x in files.values())))
    result["scientific_manifest"]=ref(manifest);write(out/"r29_summary.json",result);write(out/"r29_checks.json",result["checks"])
    write(out/"r29_receipt.json",dict(collection_commit=commit(),summary_sha256=sha(out/"r29_summary.json"),scientific_manifest=ref(manifest),
        evidence={str(p.relative_to(root)):sha(p) for p in root.rglob("*.json")},
        receipts={folder:{p.name:sha(p) for p in Path(folder).iterdir() if p.is_file()} for folder in receipts},
        launcher_logs={str(p):sha(p) for p in Path("outputs").glob(root.name+"*_launcher.log")},temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,whole_song_generations=0,operational_qualified=False))
    print("R29_PUBLIC_STATS",sha(out/"r29_summary.json"),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",required=True,choices=["verify","render","collect"]);p.add_argument("--root",required=True);p.add_argument("--out");p.add_argument("--receipts",nargs="*");a=p.parse_args()
    if a.task=="verify":verify(a.root)
    elif a.task=="render":render(a.root,a.out)
    else:collect(a.root,a.out,a.receipts)
