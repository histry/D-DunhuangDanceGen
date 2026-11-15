"""Exact replay checks, fixed-camera primary comparisons and public metadata only."""
import argparse,json,math,subprocess
from pathlib import Path
import numpy as np
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.context_r21.data import same_audio_pcm
from render_r23 import video

def verify(root):
    root=Path(root);out=root/"independent_checks.json";assert not out.exists();forbid_optimizers();bundles=0
    a=read(root/"data/result.json");b=read(root/"data_replay/result.json");assert a==b
    for folder in ["data","data_replay"]:
        for name,h in read(root/folder/"protocol.json")["files"].items():assert sha(root/folder/name)==h
    for row in a["records"]:
        same_arrays(root/"data"/row["arrays"],root/"data_replay"/row["arrays"]);bundles+=1
        assert same_audio_pcm(root/"data"/row["crop_audio"],root/"data_replay"/row["crop_audio"])
    models=[]
    for arm in ("event","phase"):
        main=read(root/f"models/{arm}/result.json");other=read(root/f"models/{arm}_replay/result.json")
        assert main["initial_digest"]==other["initial_digest"] and main["final_digest"]==other["final_digest"]
        assert main["observations"]==other["observations"] and main["curves"]==other["curves"] and main["ledger"]==other["ledger"]
        assert main["optimizer_steps"]==600 and other["optimizer_steps"]==0
        assert all(o["update_norm"]>0 for o in main["observations"])
        assert main["observations"][0]["gradient_norms"]["projections"]>0
        assert any(o["gradient_norms"]["joint"]>0 for o in main["observations"][1:])
        for p in (root/f"models/{arm}").glob("curve_*.npz"):same_arrays(p,root/f"models/{arm}_replay"/p.name);bundles+=1
        models.append(main)
    assert models[0]["initial_digest"]==models[1]["initial_digest"] and models[0]["ledger"]==models[1]["ledger"]
    assert models[0]["parameters"]==models[1]["parameters"] and models[0]["frozen_digest"]==models[1]["frozen_digest"]
    assert models[0]["curves"][0]==models[1]["curves"][0]
    for stage in ["sanity","trained_graph","samples","evaluation"]:
        assert read(root/stage/"result.json")==read(root/(stage+"_replay")/"result.json"),stage
        for p in (root/stage).glob("*.npz"):same_arrays(p,root/(stage+"_replay")/p.name);bundles+=1
    samples=read(root/"samples/result.json");assert samples["sampler_calls"]==21 and samples["total_trajectories"]==135 and samples["null_reference_exact"]
    assert len(read(root/"evaluation/result.json")["records"])==147
    frozen=["src/dunhuang/research/branch_r22","src/dunhuang/research/local_r23","src/dunhuang/research/lodge_reference_r24","src/dunhuang/edge_backend.py","src/dunhuang/planning","src/dunhuang/sampling.py"]
    assert not subprocess.check_output(["git","diff","665cd5a6bb45eb896f50cecf99f1f688b7cdf2c2","--"]+frozen,text=True)
    write(out,dict(code_commit=commit(),array_bundles_exact=bundles,source_recordings=21,audio_pcm_exact=21,train_groups=9,dev_exposed_groups=3,
        training_optimizer_steps=1200,replay_optimizer_steps=0,actual_update_pairs_exact=8,curve_checkpoints_exact=6,
        sampler_calls=21,replay_sampler_calls=21,generated_trajectories=135,replay_generated_trajectories=135,
        frozen_R24_baseline_trajectories_reused=12,metric_records_exact=147,null_reference_sampler_exact=True,
        full_native_status="NOT_EVALUABLE",whole_song_generations=0,operational_qualified=False))
    print("R25_INDEPENDENT_REPLAY_VERIFIED",bundles,flush=True)

def render(root,out):
    root=Path(root);out=fresh(out);cfg=read(root/"data/protocol.json")["config"];parent=Path(cfg["parent"])
    old=np.load(checked(read(parent/"main/protocol.json")["arrays"]));sample=np.load(root/"samples/arrays.npz")
    parity=read(Path(cfg["paired_parent"])/"data/protocol.json")["parity"];parents=np.load(checked(parity))["parents"][:22].tolist()
    tracks={};metadata={}
    def add(key,p,group,label,audio=None,annotation=None):
        tracks[key]=p;metadata[key]=dict(group=group,label=label,audio=audio,annotation=annotation)
    cue=old["cue_mask"][:,7:].all(-1).astype(bool);notes=["R24 diagnostic cue frame" if cue[t] else "Unprompted Core" if 64<=t<160 else "Outside Core" for t in range(256)]
    for e in range(2):add("Source"+str(e),old["source22"][e],"source","Source"+str(e),None,notes)
    for seed in cfg["eval_seeds"]:
        for m in range(2):
            audio=parent/"main"/f"music{m}.wav";base_i=3*m;group=f"ChangE_s{seed}_m{m}"
            add(group+"_none",old[str(seed)+"_22"][base_i],group,"NoRef",audio,notes)
            for e in range(2):
                i=base_i+e+1;add(group+"_R24_e"+str(e),old[str(seed)+"_22"][i],group,"R24_e"+str(e),audio,notes)
                for arm in cfg["arms"]:add(group+"_"+arm+"_e"+str(e),sample[f"change_{arm}_correct_{seed}_22"][i],group,arm+"_e"+str(e),audio,notes)
    pop=read(root/"data/result.json");by={r["case"]:r for r in pop["records"]}
    for case in [0,24]:
        row=by[case];z=np.load(root/"data"/row["arrays"]);group=f'AIST_{row["role"]}_case{case}';i=cfg["paired_eval_cases"].index(case);audio=root/"data"/row["crop_audio"]
        add(group+"_source",z["source22"],group,"Source",audio,notes)
        for key,label in [("baseline_none","NoRef"),("event_correct","event"),("phase_correct","phase"),("phase_occluded","phase_occluded")]:
            add(group+"_"+label,sample["paired_"+key+"_22"][i],group,label,audio,notes)
    assert len(tracks)==40
    c,s=math.cos(-.45),math.sin(-.45)
    def project(x):return np.stack((c*x[...,0]+s*x[...,2],x[...,1]-.18*(-s*x[...,0]+c*x[...,2])),-1)
    # Fixed world scale across all primary conditions; no frame-wise recentering.
    xy=np.stack([project(v) for v in tracks.values()]);lo=xy.min((0,1,2));hi=xy.max((0,1,2));center=(lo+hi)/2
    scale=min(540/max(float(hi[0]-lo[0]),1e-6),440/max(float(hi[1]-lo[1]),1e-6));media=[]
    for key,points in tracks.items():
        m=metadata[key];p=out/(key+".mp4");video(p,project(points),parents,"R25 | "+key,None,center,scale,m["audio"],m["annotation"])
        media.append(dict(key=key,group=m["group"],label=m["label"],file=p.name,video=ref(p),frames=256,full_decode_passed=True))
        print("R25_VIDEO",key,flush=True)
    page="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>R25 Lodge参考适配器</title>
<style>body{margin:24px;font:16px system-ui;background:#f2f4f7;color:#173249}select,button{margin:6px;padding:8px}.pair{display:flex;gap:20px}video{width:48%}p{max-width:1100px;line-height:1.6}</style>
<h1>R25：真实训练的完整事件／阶段适配器</h1><p>event和phase各600步，Lodge主干冻结。新模型仅内部条件控制，不向采样状态写入姿态；R24保留为稀疏写回基线。
Source0／1为飞天／琵琶来源展示，Root未知；Core外保持边界姿态只为展示。Core区间[64,160)，红字沿用R24评价分区，不是新模型的实际提示。
AIST是已曝光配对补全／参考机制案例，不是敦煌风格成果。全24关节Native不可评价，不平滑或逐帧居中。</p>
<label>A<select id="left"></select></label><label>B<select id="right"></select></label><button id="play">同步播放／暂停</button><p id="detail"></p>
<div class="pair"><video id="a" controls preload="metadata"></video><video id="b" controls muted preload="metadata"></video></div>
<script>const rows=PAYLOAD;const l=document.querySelector('#left'),r=document.querySelector('#right'),a=document.querySelector('#a'),b=document.querySelector('#b');
for(const x of rows){l.add(new Option(x.key,x.file));r.add(new Option(x.key,x.file));}
l.value='ChangE_s24041_m0_R24_e1.mp4';r.value='ChangE_s24041_m0_phase_e1.mp4';
function load(){a.pause();b.pause();a.src=l.value;b.src=r.value;document.querySelector('#detail').textContent=rows.find(x=>x.file===l.value).key+' / '+rows.find(x=>x.file===r.value).key+' | Native24 NOT EVALUABLE';}
l.onchange=load;r.onchange=load;load();document.querySelector('#play').onclick=()=>{if(a.paused){b.currentTime=a.currentTime;Promise.all([a.play(),b.play()]).catch(()=>{});}else{a.pause();b.pause();}};
function tick(){if(!a.paused&&!b.paused&&Math.abs(a.currentTime-b.currentTime)>.08)b.currentTime=a.currentTime;requestAnimationFrame(tick);}tick();</script></html>"""
    (out/"index.html").write_text(page.replace("PAYLOAD",json.dumps(media)),encoding="utf8")
    write(out/"protocol.json",dict(code_commit=commit(),records=media,live_controls_verified=False,motion_smoothing=False,fixed_world_camera=True))

def collect(root,receipts,media_receipts,out,failed_receipts):
    root=Path(root);receipts=Path(receipts);media_receipts=Path(media_receipts);out=Path(out);assert not out.exists(),"Preserve previous publication"
    for folder in (receipts,media_receipts):assert (folder/"exit_status.txt").read_text().strip()=="0" and (folder/"phase.txt").read_text().strip()=="completed"
    result=dict(data=read(root/"data/result.json"),models={a:read(root/f"models/{a}/result.json") for a in ("event","phase")},
        initial_graph=read(root/"sanity/result.json"),trained_graph=read(root/"trained_graph/result.json"),sampling=read(root/"samples/result.json"),
        evaluation=read(root/"evaluation/result.json"),checks=read(root/"independent_checks.json"),media=read(root/"videos/protocol.json"),
        human_reviewed=False,operational_qualified=False,whole_song_generations=0)
    out.mkdir(parents=True);write(out/"r25_summary.json",result);write(out/"r25_checks.json",result["checks"])
    folders=[receipts,media_receipts]+[Path(p) for p in failed_receipts]
    write(out/"r25_receipt.json",dict(collection_commit=commit(),execution_commit=(receipts/"execution_commit.txt").read_text().strip(),
        summary_sha256=sha(out/"r25_summary.json"),evidence={str(p.relative_to(root)):sha(p) for p in root.rglob("*.json") if p.name in ("protocol.json","result.json","independent_checks.json")},
        checkpoint_hashes={arm:result["models"][arm]["checkpoint_hashes"] for arm in ("event","phase")},
        receipts={str(folder):{p.name:sha(p) for p in folder.iterdir() if p.is_file()} for folder in folders},
        exit_status=0,training_optimizer_steps=1200,replay_optimizer_steps=0,temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,
        operational_qualified=False,whole_song_generations=0))
    print("R25_STATISTICS_PUBLISHED",sha(out/"r25_summary.json"),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",choices=["verify","render","collect"],required=True);p.add_argument("--root",required=True);p.add_argument("--out");p.add_argument("--receipts");p.add_argument("--media-receipts");p.add_argument("--failed-receipts",nargs="*",default=[]);a=p.parse_args()
    if a.task=="verify":verify(a.root)
    elif a.task=="render":render(a.root,a.out)
    else:collect(a.root,a.receipts,a.media_receipts,a.out,a.failed_receipts)
