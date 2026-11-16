"""Finite R32 independent replay and comparison media, without promoting research weights."""
import argparse,json,math,subprocess,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from dunhuang.io import sha,write,checked
from dunhuang.data.sources import ref
from dunhuang.research.content_r32.common import population,runtime,read,core_body,core_points
from dunhuang.research.state_r27.common import fresh,commit,forbid_optimizers,same_arrays,readout_check
from release_r29 import video

def verify(root):
    root=Path(root);out=root/"independent_checks.json";assert not out.exists();forbid_optimizers()
    cfg=read("src/dunhuang/research/content_r32/config.json");population(cfg);exact=0;checks=[];decoder_steps=0;generator_steps=0;calls=0;trajectories=0
    for stage in ("data","readout"):
        a=read(root/stage/"result.json");b=read(root/(stage+"_replay")/"result.json")
        if stage=="data":assert a==b;calls+=a["sampler_calls"];trajectories+=a["generated_trajectories"]
        else:
            for name,m in a["models"].items():
                r=b["models"][name];assert m["initial_digest"]==r["initial_digest"] and m["final_digest"]==r["final_digest"] and m["curves"]==r["curves"]
                assert m["optimizer_steps"]==600 and r["optimizer_steps"]==0
                for x,y in zip(r["actual_readouts"],m["observations"]):checks.append(readout_check(x,y))
                for fn,h in m["files"].items():assert sha(root/stage/name/fn)==h
            decoder_steps=a["decoder_optimizer_steps"];assert b["decoder_optimizer_steps"]==0 and a["generator_optimizer_steps"]==0
            assert a["partial_training_readout_admitted"] and not a["encoding_repair_indicated"]
        for p in (root/stage).glob("*.npz"):same_arrays(p,root/(stage+"_replay")/p.name);exact+=1
    for arm in ("pilot","AIST","ChangE"):
        folder=root/"models"/arm;a=read(folder/"result.json");b=read(root/"models"/(arm+"_replay")/"result.json")
        assert a["admitted"] and a["reference_parameters"]==265328 and a["action_io_trainable_parameters"]==0
        for key in ("initial_digest","final_digest","final_checkpoint_sha256","frozen","curves","history","pilot_gate"):
            assert a[key]==b[key]
        for x,y in zip(b["actual_readouts"],a["observations"]):checks.append(readout_check(x,y))
        for fn,h in a["files"].items():assert sha(folder/fn)==h
        for p in folder.glob("*.npz"):same_arrays(p,root/"models"/(arm+"_replay")/p.name);exact+=1
        assert b["optimizer_steps"]==0 and not a["partial_source_Root_C4_truth_used"] and not a["partial_full_x0_constructed"]
        assert a["frozen"]==read(root/"data/result.json")["frozen"]
        calls+=a["sampler_calls"];trajectories+=a["generated_trajectories"];generator_steps+=a["optimizer_steps"]
    a=read(root/"models/AIST/result.json");b=read(root/"models/ChangE/result.json")
    assert a["initial_digest"]==b["initial_digest"] and a["frozen"]==b["frozen"]
    paired=lambda r:[{k:x[k] for k in ("step","cases","seed","t")} for x in r["history"] if x["task"]=="paired"]
    assert paired(a)==paired(b) and len(paired(a))==300
    assert set(Counter(c for x in paired(a) for c in x["cases"]).values())=={75}
    assert [x["seed"] for x in a["history"] if x["task"]=="partial"]==[x["seed"] for x in b["history"] if x["task"]=="partial"]
    ev=read(root/"evaluation/result.json");rep=read(root/"evaluation_replay/result.json");assert ev==rep and ev["fixed_Canonical_NoRef_exact"]
    same_arrays(root/"evaluation/arrays.npz",root/"evaluation_replay/arrays.npz");exact+=1;calls+=ev["sampler_calls"];trajectories+=ev["generated_trajectories"]
    music=read(root/"music_probe/result.json");assert music==read(root/"music_probe_replay/result.json")
    same_arrays(root/"music_probe/arrays.npz",root/"music_probe_replay/arrays.npz");exact+=1;calls+=music["sampler_calls"];trajectories+=music["generated_trajectories"]
    for oldroot in (Path("outputs/geometry_r31_02"),):
        m=read(oldroot/"artifacts_manifest.json")
        for name,x in m["files"].items():assert sha(oldroot/name)==x["sha256"]
    lineage=[]
    for key,entry in read("outputs/lodge_reference_r24_01/main/protocol.json")["plans"].items():
        plan=read(checked(entry));ev=plan["events"][0];checked(ev["reference"])
        lineage.append(dict(plan=key,plan_ref=entry,**{k:ev[k] for k in ("event_uid","source_uid","recording_uid","source_sha256","source_interval","time_warp","root_translation_observed","contact_observed")}))
    assert len({x["source_sha256"] for x in lineage})==2 and len({x["recording_uid"] for x in lineage})==2
    changed=subprocess.check_output(["git","diff",cfg["baseline"],"--name-only","--","src/dunhuang"],text=True).splitlines()
    assert all(x.startswith("src/dunhuang/research/content_r32/") for x in changed)
    assert exact==13 and len(checks)==23 and decoder_steps==2400 and generator_steps==1320 and calls==53 and trajectories==236
    write(out,dict(code_commit=commit(),array_bundles_exact=exact,actual_update_pairs=23,update_checks=checks,
        decoder_optimizer_steps=decoder_steps,generator_optimizer_steps=generator_steps,replay_optimizer_steps=0,
        sampler_calls=calls,generated_trajectories=trajectories,replay_sampler_calls=calls,replay_generated_trajectories=trajectories,
        reused_evaluation_trajectories=246,fixed_NoRef_exact=True,action_io_and_backbone_unchanged=True,
        parent_scientific_seal_files_unchanged=True,paired_replay_ledger_identical=True,paired_presentations_per_source=75,
        auxiliary_sources=dict(AIST=4,ChangE=2),change_source_lineage=lineage,source_independence_scope="TWO_DISTINCT_HASHES_AND_REGISTERED_RECORDING_UIDS; NOT_NEW_OR_POPULATION_GENERALIZATION",full_native_status="NOT_EVALUABLE",operational_qualified=False,whole_song_generations=0))
    print("R32_INDEPENDENT_CHECKS",exact,23,calls,trajectories,flush=True)

def render(root,out):
    import torch
    root=Path(root);out=fresh(out);cfg=read("src/dunhuang/research/content_r32/config.json");pop,ledger,data,z=population(cfg)
    d,n,s,f,e,info,b=runtime(cfg);b.close();forbid_optimizers()
    ev=np.load(root/"evaluation/arrays.npz");prior=np.load(Path(cfg["r31"])/"evaluation/arrays.npz")
    old=np.load(Path(cfg["r31"])/"change/arrays.npz");decoded=np.load(root/"readout/arrays.npz")
    decoder_pop=[r for r in pop if r["domain"]=="aist"]+[r for r in pop if r["domain"]=="change"]
    parents=np.load(checked(read("outputs/branch_r22_03/data/protocol.json")["parity"]))["parents"][:22].tolist()
    tracks={};audio={};notes={};source_pop=read("outputs/contract_r28_01/data/result.json")["records"]
    for event in range(2):
        r=next(x for x in decoder_pop if x["uid"]=="change_"+str(event));p=core_points(r["data"]);origin=p[0,0]
        key=f"ChangE_Source{event}_RootUnknown";tracks[key]=p;notes[key]="OBSERVED SOURCE | ROOT UNKNOWN | DISPLAY HOLD OUTSIDE CORE"
        index=next(i for i,x in enumerate(decoder_pop) if x["uid"]==r["uid"])
        for view in ("raw","memory"):
            body=decoded[f"change_{view}_600"][index];raw=torch.zeros(1,96,139,device="cuda");raw[0,:,7:]=torch.tensor(body,device="cuda").clamp(-1,1)
            with torch.no_grad():points=f(raw,s)[0,:,:22].cpu().numpy()
            key=f"ChangE_{view}_readout{event}";tracks[key]=points-points[:,:1]+origin;notes[key]="READOUT ONLY | NOT DIFFUSION | ROOT UNKNOWN | DISPLAY HOLDS"
    for m in range(2):
        wav=Path(f"outputs/lodge_reference_r24_01/main/music{m}.wav")
        for event in range(2):
            i=3*m+event+1
            for arm in ("CanonicalIO","AIST","ChangE"):
                key=f"ChangE_m{m}_e{event}_{arm}";tracks[key]=(old["CanonicalIO_correct_22"] if arm=="CanonicalIO" else ev[f"change_{arm}_correct_22"])[i];audio[key]=wav;notes[key]="GENERATED | EXPOSED UNPAIRED MUSIC | ROOT NOT SOURCE TRUTH"
            for mode in ("wrong_clock","swapped"):
                key=f"ChangE_m{m}_e{event}_{mode}";tracks[key]=ev[f"change_ChangE_{mode}_22"][i];audio[key]=wav;notes[key]="GENERATED COUNTERFACTUAL | "+mode
        key=f"ChangE_m{m}_FixedNoRef";tracks[key]=old["CanonicalIO_correct_22"][3*m];audio[key]=wav;notes[key]="FIXED R31 NOREF | EXACTLY PRESERVED"
    for role,case,index in (("train",48,1),("holdout",4,0)):
        prefix=f"AIST_{role}_case{case}";src=next(x for x in source_pop if x["case"]==case);wav=Path("outputs/reference_r26_01/data")/src["source"]["crop_audio"]
        tracks[prefix+"_Source"]=data[case]["model_points"];audio[prefix+"_Source"]=wav;notes[prefix+"_Source"]="OBSERVED AIST | HISTORICAL EXPOSED SOURCE"
        for arm in ("CanonicalIO","AIST","ChangE"):
            key=prefix+"_"+arm;tracks[key]=(prior[f"{role}_CanonicalIO_correct_0_22"] if arm=="CanonicalIO" else ev[f"{role}_{arm}_correct_0_22"])[index];audio[key]=wav;notes[key]="GENERATED | SAME MUSIC / SEED / ACTION IO"
    music_probe=np.load(root/"music_probe/arrays.npz")
    for arm in ("CanonicalIO","AIST","ChangE"):
        for m in range(2):
            key=f"MatchedNoiseMusic_m{m}_e0_{arm}";tracks[key]=music_probe[arm+"_22"][3*m+1];audio[key]=Path(f"outputs/lodge_reference_r24_01/main/music{m}.wav");notes[key]="MUSIC INTERVENTION | SAME NOISE ACROSS TWO MUSIC | NOT COORDINATION TRUTH"
    assert len(tracks)==42 and len(audio)==36
    c,sn=math.cos(-.45),math.sin(-.45)
    def project(p):return np.stack((c*p[...,0]+sn*p[...,2],p[...,1]-.18*(-sn*p[...,0]+c*p[...,2])),-1)
    xy=np.concatenate([project(p).reshape(-1,2) for p in tracks.values()],0);lo=xy.min(0);hi=xy.max(0);center=(lo+hi)/2
    scale=min(540/max(float(hi[0]-lo[0]),1e-6),440/max(float(hi[1]-lo[1]),1e-6));rows=[]
    for key,p in tracks.items():
        display=len(p)==96
        if display:p=np.concatenate((np.repeat(p[:1],64,0),p,np.repeat(p[-1:],96,0)),0)
        assert len(p)==256
        labels=[notes[key]+(" | Core64:160" if 64<=t<160 else (" | DISPLAY HOLD ONLY" if display else " | Outside Core")) for t in range(256)]
        path=out/(key+".mp4");video(path,project(p),parents,"R32 | "+key,None,center,scale,audio.get(key),labels)
        rows.append(dict(key=key,file=path.name,video=ref(path),frames=256,has_audio=key in audio,full_decode_passed=True,display_holds_only=display,notes=notes[key]));print("R32_VIDEO",key,flush=True)
    page="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>R32 ChangE内容保持对照</title>
<style>body{font:16px system-ui;margin:24px;background:#f3f4f8;color:#163249}select,button{padding:8px;margin:5px}video{width:48%}p{line-height:1.7}</style>
<h1>R32：冻结动作I/O后的来源内容保持</h1><p>CanonicalIO=R31固定平台；AIST=回放+部分AIST辅助；ChangE=相同回放+部分ChangE辅助。两主臂各600步、只训练参考支路，NoRef逐位固定。Source是原观察，raw/memory readout是600步有限读出而不是扩散生成。两原语已用于适配，不是新来源盲测。</p>
<p>所有生成采用相同世界相机和比例，无平滑、重居中或贴地。Source/readout只有96帧Core，外侧仅显示定格并标DISPLAY HOLD；未知Root只在展示中置于相同来源原点，不是真实世界轨迹。Native24 NOT EVALUABLE，研究候选未晋升，整曲0。右侧静音。</p>
<select id="left"></select><select id="right"></select><button id="play">同步播放／暂停</button><p id="detail"></p><video id="a" controls preload="metadata"></video><video id="b" controls muted preload="metadata"></video>
<script>const rows=PAYLOAD,l=document.querySelector('#left'),r=document.querySelector('#right'),a=document.querySelector('#a'),b=document.querySelector('#b');
for(const x of rows){l.add(new Option(x.key,x.file));r.add(new Option(x.key,x.file));}
l.value='ChangE_m0_e0_CanonicalIO.mp4';r.value='ChangE_m0_e0_ChangE.mp4';
function load(){a.pause();b.pause();a.src=l.value;b.src=r.value;document.querySelector('#detail').textContent=l.value+' / '+r.value+' | Native24 NOT EVALUABLE';}
l.onchange=load;r.onchange=load;load();document.querySelector('#play').onclick=()=>{if(a.paused){b.currentTime=a.currentTime;Promise.all([a.play(),b.play()]).catch(()=>{});}else{a.pause();b.pause();}};
function tick(){if(!a.paused&&!b.paused&&Math.abs(a.currentTime-b.currentTime)>.08)b.currentTime=a.currentTime;requestAnimationFrame(tick);}tick();</script></html>"""
    (out/"index.html").write_text(page.replace("PAYLOAD",json.dumps(rows)),encoding="utf8")
    write(out/"protocol.json",dict(code_commit=commit(),records=rows,fixed_world_camera=True,motion_smoothing=False,generated_motion_ground_refit=False,live_browser_controls_verified=False))

def collect(root,out,receipts):
    root=Path(root);out=fresh(out)
    for name in receipts:
        p=Path(name);assert (p/"exit_status.txt").read_text().strip()=="0" and (p/"phase.txt").read_text().strip()=="completed"
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.rglob("*") if p.is_file()}
    manifest=root/"artifacts_manifest.json";assert not manifest.exists();write(manifest,dict(code_commit=commit(),files=files,file_count=len(files),total_bytes=sum(x["bytes"] for x in files.values())))
    models={}
    for arm in ("pilot","AIST","ChangE"):
        a=read(root/f"models/{arm}/result.json");models[arm]={k:v for k,v in a.items() if k!="history"};models[arm]["history_sha256"]=hashlib.sha256(json.dumps(a["history"],sort_keys=True).encode()).hexdigest();models[arm]["replay_checks"]=read(root/f"models/{arm}_replay/result.json")["replay_checks"]
    summary=dict(config=read("src/dunhuang/research/content_r32/config.json"),data=read(root/"data/result.json"),readout=read(root/"readout/result.json"),
        models=models,evaluation=read(root/"evaluation/result.json"),music_probe=read(root/"music_probe/result.json"),checks=read(root/"independent_checks.json"),media=read(root/"videos/protocol.json"),
        scientific_manifest=ref(manifest),human_reviewed=False,operational_qualified=False,whole_song_generations=0,full_native_status="NOT_EVALUABLE")
    write(out/"r32_summary.json",summary);write(out/"r32_checks.json",summary["checks"])
    write(out/"r32_receipt.json",dict(collection_commit=commit(),summary_sha256=sha(out/"r32_summary.json"),scientific_manifest=ref(manifest),
        receipts={name:{p.name:sha(p) for p in Path(name).iterdir() if p.is_file()} for name in receipts},
        evidence={str(p.relative_to(root)):sha(p) for p in root.rglob("*.json")},
        launcher_logs={str(p):sha(p) for p in Path("outputs").glob("content_r32_*_launcher.log")},
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False,whole_song_generations=0))
    print("R32_PUBLIC_STATS",sha(out/"r32_summary.json"),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",choices=["verify","render","collect"],required=True);p.add_argument("--root",required=True);p.add_argument("--out");p.add_argument("--receipts",nargs="*");a=p.parse_args()
    if a.task=="verify":verify(a.root)
    elif a.task=="render":render(a.root,a.out)
    else:collect(a.root,a.out,a.receipts)
