"""Read-only independent seals, shared-camera videos, compact public provenance."""
import argparse,json,math,subprocess,hashlib
from pathlib import Path
import numpy as np
from dunhuang.io import sha,write,checked
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.context_r21.data import same_audio_pcm
from dunhuang.research.reference_r26.common import check_readout
from render_r23 import video

def verify(root):
    root=Path(root);out=root/"independent_checks.json";assert not out.exists();forbid_optimizers();bundles=0
    data=read(root/"data/result.json");assert data==read(root/"data_replay/result.json")
    for folder in ("data","data_replay"):
        for name,h in read(root/folder/"protocol.json")["files"].items():assert sha(root/folder/name)==h
    for row in data["records"]:
        same_arrays(root/"data"/row["arrays"],root/"data_replay"/row["arrays"]);bundles+=1
        assert same_audio_pcm(root/"data"/row["crop_audio"],root/"data_replay"/row["crop_audio"])
    models=[];max_gradient=0.
    for arm in ("C0","C1","C2"):
        a=read(root/f"models/{arm}/result.json");b=read(root/f"models/{arm}_replay/result.json")
        for name,h in a["checkpoint_hashes"].items():assert sha(root/f"models/{arm}"/name)==h
        assert a["optimizer_steps"]==1200 and b["optimizer_steps"]==0
        assert a["initial_digest"]==b["initial_digest"] and a["final_digest"]==b["final_digest"]
        assert a["ledger"]==b["ledger"] and a["curves"]==b["curves"]
        for old,actual in zip(a["observations"],b["actual_replay_readouts"]):
            result=check_readout(actual,old);max_gradient=max(max_gradient,max(x["relative_error"] for x in result["deltas"].values()))
        assert len(a["observations"])==len(b["actual_replay_readouts"])==4
        assert a["observations"][0]["gradient_norms"]["projections"]>0 and a["observations"][-1]["gradient_norms"]["encoder"]>0
        assert all(x["update_norm"]>0 for x in a["observations"])
        for p in (root/f"models/{arm}").glob("curve_*.npz"):same_arrays(p,root/f"models/{arm}_replay"/p.name);bundles+=1
        models.append(a)
    assert models[0]["ledger"]==models[1]["ledger"]==models[2]["ledger"]
    assert models[1]["initial_digest"]==models[2]["initial_digest"]
    assert models[0]["frozen_digest"]==models[1]["frozen_digest"]==models[2]["frozen_digest"]
    for stage in ("preflight","compress","samples","evaluation"):
        assert read(root/stage/"result.json")==read(root/(stage+"_replay")/"result.json")
        for p in (root/stage).glob("*.npz"):same_arrays(p,root/(stage+"_replay")/p.name);bundles+=1
    assert read(root/"samples/result.json")["sampler_calls"]==26 and read(root/"samples/result.json")["generated_trajectories"]==166
    assert len(read(root/"evaluation/result.json")["records"])==202
    frozen=["src/dunhuang/research/lodge_adapter_r25","src/dunhuang/research/branch_r22","src/dunhuang/research/local_r23","src/dunhuang/research/lodge_reference_r24","src/dunhuang/edge_backend.py","src/dunhuang/planning","src/dunhuang/sampling.py"]
    assert not subprocess.check_output(["git","diff","046c52605f53154f737c43bc7d895d1365771592","--"]+frozen,text=True)
    write(out,dict(code_commit=commit(),source_recordings=102,Train_recordings=96,DEV_exposed_recordings=6,audio_pcm_exact=102,array_bundles_exact=bundles,
        training_optimizer_steps=3600,replay_optimizer_steps=0,actual_update_pairs_verified=12,curve_checkpoints_exact=9,
        gradient_norm_rtol=1e-5,gradient_norm_atol=0,maximum_gradient_norm_relative_error=max_gradient,non_gradient_readouts_exact=True,
        sampler_calls=26,replay_sampler_calls=26,generated_trajectories=166,replay_generated_trajectories=166,metric_records_exact=202,
        C3_training_optimizer_steps=0,whole_song_generations=0,full_native_status="NOT_EVALUABLE",operational_qualified=False))
    print("R26_INDEPENDENT_CHECKS",bundles,flush=True)

def render(root,out):
    root=Path(root);out=fresh(out);cfg=read(root/"data/protocol.json")["config"]
    r24=Path(cfg["r24"]);old=np.load(checked(read(r24/"main/protocol.json")["arrays"]))
    r25=np.load(Path(cfg["r25"])/"samples/arrays.npz");new=np.load(root/"samples/arrays.npz")
    parity=read("outputs/branch_r22_03/data/protocol.json")["parity"];parents=np.load(checked(parity))["parents"][:22].tolist()
    tracks={};meta={}
    cue=old["cue_mask"][:,7:].all(-1).astype(bool)
    notes=["R24 diagnostic frame; R26 has no state write" if cue[t] else "Reference Core" if 64<=t<160 else "Outside Core" for t in range(256)]
    def add(key,p,audio=None):
        tracks[key]=p;meta[key]=dict(audio=audio)
    for e in range(2):add(f"Source{e}",old["source22"][e])
    for seed in cfg["eval_seeds"]:
        for m in range(2):
            group=f"ChangE_s{seed}_m{m}";audio=r24/"main"/f"music{m}.wav";add(group+"_none",old[f"{seed}_22"][3*m],audio)
            for e in range(2):
                i=3*m+e+1;add(group+f"_R25phase_e{e}",r25[f"change_phase_correct_{seed}_22"][i],audio)
                for arm in cfg["arms"]:add(group+f"_{arm}_e{e}",new[f"change_{arm}_correct_{seed}_22"][i],audio)
    pop=read(root/"data/result.json");case=pop["dev_probe"][0];i=(pop["train_probe"]+pop["dev_probe"]).index(case)
    row=next(r for r in pop["records"] if r["case"]==case);source=np.load(root/"data"/row["arrays"])
    audio=root/"data"/row["crop_audio"];group=f"AIST_dev_case{case}";add(group+"_Source",source["source22"],audio)
    for stem,label in [("paired_baseline_none","none"),("paired_C1_correct","C1"),("paired_C2_correct","C2")]:add(group+"_"+label,new[stem+"_22"][i],audio)
    assert len(tracks)==42
    c,s=math.cos(-.45),math.sin(-.45)
    def project(x):return np.stack((c*x[...,0]+s*x[...,2],x[...,1]-.18*(-s*x[...,0]+c*x[...,2])),-1)
    xy=np.stack([project(v) for v in tracks.values()]);lo=xy.min((0,1,2));hi=xy.max((0,1,2));center=(lo+hi)/2
    scale=min(540/max(float(hi[0]-lo[0]),1e-6),440/max(float(hi[1]-lo[1]),1e-6));records=[]
    for key,points in tracks.items():
        p=out/(key+".mp4");video(p,project(points),parents,"R26 | "+key,None,center,scale,meta[key]["audio"],notes)
        records.append(dict(key=key,file=p.name,video=ref(p),frames=256,full_decode_passed=True))
        print("R26_VIDEO",key,flush=True)
    page="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>R26完整来源构形对照</title>
<style>body{font:16px system-ui;background:#f3f4f8;color:#163249;margin:24px}select,button{padding:8px;margin:5px}video{width:48%}p{line-height:1.7}</style>
<h1>R26：有序关节编码与配对几何监督</h1><p>C0原部位均值，C1有序关节编码，C2相同编码加几何/速度残差。三组各1200步、共同96Train/6曝光DEV。Lodge原音乐与主干冻结。历史R25 phase单独标记。</p>
<p>此页是22关节研究结果，不是已通过敦煌风格或全Native的成果。Source仅展示原语Core，Core外持姿不是生成目标；三组均无采样姿态写回。固定世界相机，无平滑/逐帧居中。只播放左侧音频。</p>
<select id="left"></select><select id="right"></select><button id="play">同步播放/暂停</button><p id="detail"></p><video id="a" controls></video><video id="b" controls muted></video>
<script>const rows=PAYLOAD,l=document.querySelector('#left'),r=document.querySelector('#right'),a=document.querySelector('#a'),b=document.querySelector('#b');
for(const x of rows){l.add(new Option(x.key,x.file));r.add(new Option(x.key,x.file));}
l.value='ChangE_s24041_m0_C0_e1.mp4';r.value='ChangE_s24041_m0_C2_e1.mp4';
function load(){a.pause();b.pause();a.src=l.value;b.src=r.value;document.querySelector('#detail').textContent=l.value+' / '+r.value+' | Native24 NOT EVALUABLE';}
l.onchange=load;r.onchange=load;load();document.querySelector('#play').onclick=()=>{if(a.paused){b.currentTime=a.currentTime;Promise.all([a.play(),b.play()]).catch(()=>{});}else{a.pause();b.pause();}};
function tick(){if(!a.paused&&!b.paused&&Math.abs(a.currentTime-b.currentTime)>.08)b.currentTime=a.currentTime;requestAnimationFrame(tick);}tick();</script></html>"""
    (out/"index.html").write_text(page.replace("PAYLOAD",json.dumps(records)),encoding="utf8")
    write(out/"protocol.json",dict(code_commit=commit(),records=records,motion_smoothing=False,fixed_world_camera=True,browser_interaction_tested=False))

def collect(root,out,receipts):
    root=Path(root);out=fresh(out)
    for folder in receipts:
        p=Path(folder);assert (p/"exit_status.txt").read_text().strip()=="0" and (p/"phase.txt").read_text().strip()=="completed"
    models={}
    for arm in ("C0","C1","C2"):
        a=read(root/f"models/{arm}/result.json");b=read(root/f"models/{arm}_replay/result.json")
        models[arm]={k:v for k,v in a.items() if k not in ("history","ledger")}
        for k in ("history","ledger"):models[arm][k+"_sha256"]=hashlib.sha256(json.dumps(a[k],sort_keys=True).encode()).hexdigest()
        models[arm]["original_result"]=ref(root/f"models/{arm}/result.json")
        models[arm]["replay_checks"]=b["replay_checks"]
    result=dict(data=read(root/"data/result.json"),preflight=read(root/"preflight/result.json"),models=models,compress=read(root/"compress/result.json"),
        samples=read(root/"samples/result.json"),evaluation=read(root/"evaluation/result.json"),checks=read(root/"independent_checks.json"),media=read(root/"videos/protocol.json"),
        human_reviewed=False,operational_qualified=False,whole_song_generations=0)
    write(out/"r26_summary.json",result);write(out/"r26_checks.json",result["checks"])
    write(out/"r26_receipt.json",dict(collection_commit=commit(),summary_sha256=sha(out/"r26_summary.json"),
        evidence={str(p.relative_to(root)):sha(p) for p in root.rglob("*.json") if p.name in ("result.json","protocol.json","independent_checks.json") or p.name.startswith("actual_")},
        receipts={folder:{p.name:sha(p) for p in Path(folder).iterdir() if p.is_file()} for folder in receipts},
        launcher_logs={str(p):sha(p) for folder in receipts for p in [Path(folder.removesuffix("_receipts")+"_launcher.log")] if p.exists()},
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,training_optimizer_steps=3600,replay_optimizer_steps=0,operational_qualified=False,whole_song_generations=0))
    print("R26_PUBLIC_STATS",sha(out/"r26_summary.json"),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",required=True,choices=["verify","render","collect"]);p.add_argument("--root",required=True);p.add_argument("--out");p.add_argument("--receipts",nargs="*");a=p.parse_args()
    if a.task=="verify":verify(a.root)
    elif a.task=="render":render(a.root,a.out)
    else:collect(a.root,a.out,a.receipts)
