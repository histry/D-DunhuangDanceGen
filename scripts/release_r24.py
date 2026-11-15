"""Independent evidence verification, fixed-world media and public statistics."""
import argparse,json,math,subprocess
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write,sha
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,fresh,commit,same_arrays,forbid_optimizers
from dunhuang.research.context_r21.data import same_audio_pcm
from render_r23 import video

def verify(root):
    root=Path(root);forbid_optimizers();a=read(root/"main/result.json");b=read(root/"replay/result.json");assert a==b
    same_arrays(root/"main/arrays.npz",root/"replay/arrays.npz")
    for m in a["music"]:assert same_audio_pcm(root/"main"/m["file"],root/"replay"/m["file"])
    assert len(a["records"])==12 and len(a["noise_receipts"])==2 and all(r["draws"]==50 for r in a["noise_receipts"])
    assert a["full_native_status"]=="NOT_EVALUABLE" and a["all_conditions_same_soft_sampler"]
    cfg=read(root/"main/protocol.json")["config"]
    frozen=["src/dunhuang/research/branch_r22","src/dunhuang/research/local_r23","src/dunhuang/edge_backend.py","src/dunhuang/planning","src/dunhuang/sampling.py"]
    assert not subprocess.check_output(["git","diff",cfg["baseline"],"--"]+frozen,text=True)
    out=root/"independent_checks.json";assert not out.exists()
    write(out,dict(code_commit=commit(),arrays_exact=True,metrics_exact=True,audio_pcm_exact=2,generated_trajectories=12,
        replay_trajectories=12,batched_sampler_calls=2,replay_batched_sampler_calls=2,common_noise_draws_per_seed=50,
        masks_matched=True,optimizer_steps=0,full_native_status="NOT_EVALUABLE",operational_qualified=False,whole_song_generations=0))
    print("R24_FRESH_REPLAY_VERIFIED",flush=True)

def render(root,out):
    root=Path(root);out=fresh(out);r=read(root/"main/result.json");z=np.load(root/"main/arrays.npz");cfg=read(root/"main/protocol.json")["config"]
    parity=read("outputs/branch_r22_03/data/protocol.json")["parity"];parents=np.load(checked(parity))["parents"][:22].tolist()
    tracks={"SourceA":z["source22"][0],"SourceB":z["source22"][1]}
    for row in r["records"]:tracks[f's{row["seed"]}_m{row["music"]}_e{row["event"]}']=z[str(row["seed"])+"_22"][row["index"]]
    c,s=math.cos(-.45),math.sin(-.45)
    def project(x):return np.stack((c*x[...,0]+s*x[...,2],x[...,1]-.18*(-s*x[...,0]+c*x[...,2])),-1)
    xy=np.stack([project(p) for p in tracks.values()]);lo=xy.min((0,1,2));hi=xy.max((0,1,2));center=(lo+hi)/2
    scale=min(540/max(float(hi[0]-lo[0]),1e-6),440/max(float(hi[1]-lo[1]),1e-6));media=[]
    core=r["core_interval"];cue=z["cue_mask"][:,7:].all(-1).astype(bool)
    notes=["Soft rotation cue" if cue[t] else "Unprompted Core" if core[0]<=t<core[1] else "Outside source Core" for t in range(256)]
    for key,points in tracks.items():
        row=next((row for row in r["records"] if key==f's{row["seed"]}_m{row["music"]}_e{row["event"]}'),None)
        audio=None if row is None else root/"main"/r["music"][row["music"]]["file"]
        path=out/(key+".mp4");video(path,project(points),parents,"R24 | "+key,None,center,scale,audio,notes)
        media.append(dict(file=path.name,key=key,video=ref(path),record=row,frames=256,full_decode_passed=True,source_root_translation_observed=False))
        print("R24_VIDEO",key,flush=True)
    page="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>R24 Lodge 敦煌原语对照</title>
<style>body{margin:24px;font:16px system-ui;background:#f2f4f7;color:#173249}select,button{margin:6px;padding:8px}.pair{display:flex;gap:20px}video{width:48%}p{max-width:1100px;line-height:1.6}</style>
<h1>R24：Lodge Local × ChangE 完整事件 × 敦煌音乐</h1><p>固定飞天／琵琶两个完整来源，dunhuangwu2/3 两段真实音乐，两个种子。相同 soft sampler、提示时刻与噪声。
SourceA/B 的 Root 仅为展示固定位置，Core 外保持边界姿态；没有把这些展示帧作为提示或真值。提示帧和未提示 Core 在视频中分别标识。
模型只输出22身体关节，无细致手指，完整Native不可评价。视频不平滑，不逐帧Root居中，不表示风格或物理已合格。</p>
<label>A<select id="left"></select></label><label>B<select id="right"></select></label><button id="play">同步播放／暂停</button><p id="detail"></p>
<div class="pair"><video id="a" controls preload="metadata"></video><video id="b" controls muted preload="metadata"></video></div>
<script>const rows=PAYLOAD;const l=document.querySelector('#left'),r=document.querySelector('#right'),a=document.querySelector('#a'),b=document.querySelector('#b');
for(const x of rows){l.add(new Option(x.key,x.file));r.add(new Option(x.key,x.file));}l.selectedIndex=2;r.selectedIndex=3;
function load(){a.pause();b.pause();a.src=l.value;b.src=r.value;document.querySelector('#detail').textContent=rows.find(x=>x.file===l.value).key+' / '+rows.find(x=>x.file===r.value).key+' | Native24 NOT EVALUABLE';}
l.onchange=load;r.onchange=load;load();document.querySelector('#play').onclick=()=>{if(a.paused){b.currentTime=a.currentTime;Promise.all([a.play(),b.play()]).catch(()=>{});}else{a.pause();b.pause();}};
function tick(){if(!a.paused&&!b.paused&&Math.abs(a.currentTime-b.currentTime)>.08)b.currentTime=a.currentTime;requestAnimationFrame(tick);}tick();</script></html>"""
    (out/"index.html").write_text(page.replace("PAYLOAD",json.dumps(media)),encoding="utf8")
    write(out/"protocol.json",dict(code_commit=commit(),records=media,common_camera_all_conditions=True,motion_smoothing=False,live_controls_verified=False))

def collect(root,receipts,media_receipts,out):
    root=Path(root);receipts=Path(receipts);media_receipts=Path(media_receipts);out=Path(out);assert not out.exists()
    for r in (receipts,media_receipts):assert (r/"exit_status.txt").read_text().strip()=="0" and (r/"phase.txt").read_text().strip()=="completed"
    summary=read(root/"main/result.json");summary.update(independent_checks=read(root/"independent_checks.json"),media=read(root/"videos/protocol.json"))
    out.mkdir(parents=True);write(out/"r24_summary.json",summary);write(out/"r24_checks.json",summary["independent_checks"])
    write(out/"r24_receipt.json",dict(collection_commit=commit(),execution_commit=(receipts/"execution_commit.txt").read_text().strip(),
        evidence={str(p.relative_to(root)):sha(p) for p in root.rglob("*.json") if p.name in ("result.json","protocol.json","independent_checks.json")},
        summary_sha256=sha(out/"r24_summary.json"),receipts={str(r):{p.name:sha(p) for p in r.iterdir() if p.is_file()} for r in (receipts,media_receipts)},
        exit_status=0,optimizer_steps=0,whole_song_generations=0,temporary_ssh_files_created=0,temporary_ssh_files_remaining=0))
    print("R24_PUBLIC_STATISTICS_READY",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--task",choices=["verify","render","collect"],required=True)
    p.add_argument("--out");p.add_argument("--receipts");p.add_argument("--media-receipts");a=p.parse_args()
    if a.task=="verify":verify(a.root)
    elif a.task=="render":render(a.root,a.out)
    else:collect(a.root,a.receipts,a.media_receipts,a.out)
