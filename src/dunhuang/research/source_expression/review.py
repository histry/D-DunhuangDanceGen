"""Private reproducible blind-review handoff; no human annotations are generated."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
from dunhuang.io import checked,sha,write,pinned
from dunhuang.data.sources import ref
from dunhuang.data.split_registry import stable
from .common import read,commit,fresh
from .review_contract import VERSION,summarize

PARENTS=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
def page(points,task,audio=None):
    # Only opaque media, geometry and neutral task instructions reach a reviewer.
    data=json.dumps(np.round(points,6).tolist(),separators=(",",":"))
    return """<!doctype html><meta charset="utf-8"><title>Independent review</title>
<style>body{font:18px sans-serif;max-width:900px;margin:30px auto}canvas{background:#eee}button,input,select{font:inherit;margin:8px}</style>
<p>Task: TASK. Inspect independently. No automatic event label or model score is provided.
For MUSIC_MATCH judge compatibility only. UNSURE is permitted. Fitted root-relative skeleton;
invisible details must be marked UNSURE. Mouse drag rotates the view.</p>
<canvas width="800" height="500"></canvas><br><button id="play">Play / pause</button>
<input id="frame" type="range" min="0" max="LAST" value="0"><span id="time"></span>AUDIO
<script>
const xyz=DATA,parents=PARENTS,c=document.querySelector('canvas'),g=c.getContext('2d');
const radius=Math.max(.01,...xyz.flat().map(p=>Math.hypot(...p))),scale=210/radius;
let f=0,playing=false,angle=0,drag=false,audio=document.querySelector('audio');
c.onpointerdown=()=>drag=true;c.onpointerup=()=>drag=false;c.onpointermove=e=>{if(drag)angle+=e.movementX*.01};
document.querySelector('#play').onclick=()=>{playing=!playing;if(audio){if(playing)audio.play();else audio.pause()}};
document.querySelector('#frame').oninput=e=>{f=+e.target.value;playing=false;if(audio){audio.pause();audio.currentTime=f/30}};
setInterval(()=>{if(audio)f=Math.min(xyz.length-1,Math.floor(audio.currentTime*30));else if(playing)f=(f+1)%xyz.length;
g.clearRect(0,0,800,500);const pts=xyz[f].map(p=>[400+scale*(Math.cos(angle)*p[0]+Math.sin(angle)*p[2]),250-scale*p[1]]);
for(let j=1;j<pts.length;j++){g.beginPath();g.moveTo(...pts[parents[j]]);g.lineTo(...pts[j]);g.stroke()}
document.querySelector('#frame').value=f;document.querySelector('#time').textContent=(f/30).toFixed(2)+' s';},1000/30);
</script>""".replace("TASK",task).replace("LAST",str(len(points)-1)).replace("AUDIO",'<audio controls src="'+audio+'"></audio>' if audio else "").replace("DATA",data).replace("PARENTS",json.dumps(PARENTS))

def run(config,prepared,out,replay_from=None):
    import soundfile as sf
    out=fresh(out);public=out/"reviewer";private=out/"administrator";public.mkdir();private.mkdir()
    prep=read(Path(prepared)/"protocol.json")
    if prep["config"]!=config or config["review"]["actual_records"] is not None:raise ValueError("Preparation-only protocol required")
    root=Path(config["data_v3"]);evidence=read("reports/data_event_v3_summary.json")["evidence"]
    for name,digest in evidence.items():
        if sha(root/name)!=digest:raise ValueError("Sealed source differs")
    parent=read(root/"main/protocol.json");assets=read(checked(parent["assets"]))
    sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    from motion.geometry import fk_24_np
    entries=read(root/"main/event_catalog_v2_shadow.json")["entries"];seals=read(root/"main/source_seals.json")
    selected=[];sources={};renderer=Path(__file__)
    for source in seals:
        checked(source["copied_raw"]);native=np.load(checked(source["original_native"]));fk=fk_24_np(native)
        sources[source["source_uid"]]=(source,native,fk-fk[:,:1])
        pool=[e for e in entries if e["source_uid"]==source["source_uid"]]
        selected.extend(sorted(pool,key=lambda e:stable([config["selection_seed"],e["event_uid"]]))[:config["review"]["events_per_source"]])
    bindings=[];items=[];music=read(checked(prep["music"]))
    tasks=[(e,task,None) for e in selected for task in ("EVENT","POSE")]
    tasks.extend((selected[i], "MUSIC_MATCH",m) for i,m in enumerate(music))
    for i,(event,task,m) in enumerate(tasks):
        source,native,fk=sources[event["source_uid"]];a,b=event["source_interval"];context=config["review"]["context_frames"]
        lo=max(0,a-context);hi=min(len(native),b+context)
        bid=hashlib.sha256((str(config["selection_seed"])+str(i)+event["event_uid"]+task).encode()).hexdigest()[:24]
        audio=None;audio_ref=None
        if m is not None:
            wave,sr=sf.read(checked(m["audio"]),dtype="float32",always_2d=True);count=round((hi-lo)*sr/30)
            if count>len(wave):raise ValueError("Real compatibility audio too short")
            audio=bid+".wav";sf.write(public/audio,wave[:count],sr,subtype="FLOAT");audio_ref=ref(public/audio)
        (public/(bid+".html")).write_text(page(fk[lo:hi,:22],task,audio),encoding="utf-8")
        media=ref(public/(bid+".html"))
        binding=dict(blind_item_id=bid,task_type=task,source_uid=event["source_uid"],raw_motion_sha256=source["raw_sha256"],
            event_uid=event["event_uid"],start_frame=a,end_frame=b,fps=30,display_interval=[lo,hi],media_sha256=media["sha256"],
            raw=source["copied_raw"],native=source["original_native"],media=media,audio=audio_ref,music_identity=m,
            source_clock="original half-open native frames; root-relative visualization only",renderer_sha256=sha(renderer))
        bindings.append(binding);items.append(dict(blind_item_id=bid,task_type=task,media=bid+".html",media_sha256=media["sha256"]))
    # Source identities and roles are NEVER placed in reviewer files.
    write(public/"items.json",items)
    (public/"README.txt").write_text("PREPARATION ONLY. Assign separate private copies to independent reviewers. Open index.html. Do not share submissions. Submit at least two independent reviews per item; third adjudication for disagreements. Administrator binds sources after review. No sample has passed.\n",encoding="utf-8")
    template=Path(__file__).with_name("review_ui.html").read_text(encoding="utf-8")
    (public/"index.html").write_text(template.replace("__ITEMS__",json.dumps(items)).replace("__VERSION__",VERSION),encoding="utf-8")
    write(private/"bindings.json",bindings);write(private/"roster_template.json",dict(reviewers={},note="Private verified identities required; never publish."))
    summary,decisions,records=summarize(bindings,[],{},"PRIVATE_ROSTER_NOT_YET_REGISTERED")
    result=dict(status="PREPARATION_ONLY",items=len(items),events=len(selected),sources=len(seals),tasks={t:sum(i["task_type"]==t for i in items) for t in ("EVENT","POSE","MUSIC_MATCH")},
        human_reviewed=False,semantic_event_truth="UNKNOWN",summary=summary,actual_records=0,fresh_blind_evaluation_population=False,
        exposure="Previously used ChangE sources; pilot handoff only. Independent untouched evaluation population NOT established.",
        media_representation="fitted 22-joint root-relative skeleton; no reliable world Root/contact or unseen anatomy claim",operational_qualified=False)
    write(private/"decisions.json",decisions);write(out/"result.json",result)
    write(out/"protocol.json",dict(code_commit=commit(),config=config,prepared=ref(Path(prepared)/"protocol.json"),bindings=ref(private/"bindings.json"),
        manifest=ref(public/"items.json"),renderer=ref(renderer),ui=ref(Path(__file__).with_name("review_ui.html")),adjudication="separate third independent original vote; preserve originals"))
    if replay_from:
        prev=Path(replay_from)
        if read(prev/"result.json")!=result or read(prev/"reviewer/items.json")!=items:raise ValueError("Review handoff differs")
        for file in public.iterdir():
            if sha(file)!=sha(prev/"reviewer"/file.name):raise ValueError("Blind media bytes differ")
        write(out/"replay.json",dict(status="FRESH_BLIND_PREPARATION_REPLAY",optimizer_calls=0,exact_media=True,actual_records=0))
    print("R6_BLIND_PREPARATION_COMPLETED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--prepared",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.prepared,a.out,a.replay_from)
