"""Fixed-world single-panel videos; synchronized selectable two-panel research viewer."""
import argparse,json,math,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.state_bridge.common import native_environment

def run(root,out):
    root=Path(root);out=fresh(out);pre=read(root/"data/protocol.json");cfg=pre["config"];population=read(root/"data/result.json")
    native_environment(pre["assets"])
    from motion.geometry import fk_24_np
    from motion_geometry.smpl24 import PARENTS
    saved=read(root/"native/protocol.json")["saved"];records=read(root/"native/result.json")["records"]
    mapping={(b["arm"],b["case"],b["variant"]):(root/"native"/folder,int(folder)) for folder,b in saved.items() if b["control"]=="correct"}
    arms=[a for a in cfg["arms"] if a!="F" or population["support_admitted"]]
    cases=[population["train_probe"][0],population["dev_cases"][0]];by={r["case"]:r for r in population["records"]};media=[];font=ImageFont.load_default()
    c,s=math.cos(-.45),math.sin(-.45)
    def project(x):return np.stack((c*x[...,0]+s*x[...,2],x[...,1]-.18*(-s*x[...,0]+c*x[...,2])),-1)
    for case in cases:
        row=by[case];z=np.load(checked(row["arrays"]));source=fk_24_np(z["native"])
        tracks={"Source":source}
        for variant in cfg["eval_variants"]:
            for arm in arms:tracks[arm+"_"+str(variant)]=np.load(mapping[arm,case,variant][0]/"fk.npy")
        xy=np.stack([project(v) for v in tracks.values()]);lo=xy.min((0,1,2));hi=xy.max((0,1,2));center=(lo+hi)/2
        scale=min(540/max(float(hi[0]-lo[0]),1e-6),440/max(float(hi[1]-lo[1]),1e-6))
        for variant in cfg["eval_variants"]:
            for arm in ["Source"]+arms:
                points=project(source if arm=="Source" else tracks[arm+"_"+str(variant)])
                path=out/f"case{case}_v{variant}_{arm}.mp4";native_ok=read(checked(row["source_physical"]))["absolute"]["ok"] if arm=="Source" else records[mapping[arm,case,variant][1]]["native_ok"]
                label=f'{row["role"]} case{case} | {arm} | context{cfg["contexts"][variant]}'
                proc=subprocess.Popen(["ffmpeg","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","600x600","-r","30","-i","-","-i",str(checked(row["audio225"])),
                    "-map","0:v:0","-map","1:a:0","-c:v","libx264","-pix_fmt","yuv420p","-preset","veryfast","-crf","20","-threads","2","-c:a","aac","-shortest",str(path)],stdin=subprocess.PIPE)
                try:
                    for t in range(225):
                        image=Image.new("RGB",(600,600),"#f2f4f7");draw=ImageDraw.Draw(image)
                        draw.text((12,12),label,fill="#173249",font=font)
                        draw.text((12,34),f'Native {"PASS" if native_ok else "FAIL"} | AIST mechanism / not Dunhuang artwork',fill="#70402e",font=font)
                        pts=[(round(300+(v[0]-center[0])*scale),round(310-(v[1]-center[1])*scale)) for v in points[t]]
                        for j in range(1,24):draw.line([pts[int(PARENTS[j])],pts[j]],fill="#285e99" if j%2 else "#9c5133",width=4)
                        for a,b in pts:draw.ellipse((a-3,b-3,a+3,b+3),fill="#22394c")
                        draw.text((12,553),f"{t+1}/225 | world Root | common fixed camera and scale",fill="#415b6b",font=font)
                        draw.text((12,576),"No frame recentering / smoothing | unqualified research",fill="#415b6b",font=font)
                        proc.stdin.write(np.asarray(image).tobytes())
                finally:proc.stdin.close()
                assert proc.wait()==0
                frames=int(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","csv=p=0",str(path)],text=True).strip())
                assert frames==225;subprocess.run(["ffmpeg","-v","error","-i",str(path),"-f","null","-"],check=True)
                media.append(dict(case=case,role=row["role"],variant=variant,arm=arm,video=ref(path),native_ok=native_ok,frames=225,full_decode_passed=True,audio=row["audio225"]))
            print("R21_VIDEO_SET",case,variant,flush=True)
    payload=[dict(x,file=Path(x["video"]["path"]).name) for x in media]
    page="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>R21 同步研究对比</title>
<style>body{margin:24px;font:16px system-ui;background:#f2f4f7;color:#173249}select,button,input{margin:6px;padding:8px}.pair{display:flex;gap:20px}.pane{width:48%;overflow:auto;background:white;border-radius:12px;padding:8px}video{display:block;width:600px;max-width:none}p{max-width:1100px;line-height:1.6}</style>
<h1>R21：边界、内部时序与支撑机制</h1>
<p>配对 AIST 机制实验。原未来双端已知；不是敦煌舞成品，不验证文化语义或未知未来 Root。
同一来源的全部模型与观察布局使用共同固定世界视角、共同显示比例，不平滑，不逐帧居中。缩放只改变显示。
此页选取预登记的首个 Train 和首个曝光 DEV，完整人口与 Native 在报告中保留。</p>
<label>来源/观察布局<select id="case"></select></label><label>A<select id="left"></select></label><label>B<select id="right"></select></label>
<button id="play">同步播放/暂停</button><label>显示缩放<input id="zoom" type="range" min="1" max="2.5" step=".25" value="1"></label>
<div class="pair"><div class="pane"><h2 id="la"></h2><video id="a" controls preload="metadata"></video></div><div class="pane"><h2 id="lb"></h2><video id="b" controls muted preload="metadata"></video></div></div>
<script>
const rows=PAYLOAD;
const a=document.querySelector('#a'),b=document.querySelector('#b'),cs=document.querySelector('#case'),ls=document.querySelector('#left'),rs=document.querySelector('#right');
const keys=[...new Set(rows.map(x=>x.case+':'+x.variant))],arms=[...new Set(rows.map(x=>x.arm))];
for(const k of keys){let r=rows.find(x=>x.case+':'+x.variant===k);cs.add(new Option(r.role+' case'+r.case+' / layout'+r.variant,k));}
for(const x of arms){ls.add(new Option(x,x));rs.add(new Option(x,x));}ls.value='Source';rs.value='B3';b.muted=true;
function load(){const key=cs.value;let l=rows.find(x=>x.case+':'+x.variant===key&&x.arm===ls.value),r=rows.find(x=>x.case+':'+x.variant===key&&x.arm===rs.value);a.pause();b.pause();a.src=l.file;b.src=r.file;document.querySelector('#la').textContent=l.arm+' / Native '+(l.native_ok?'PASS':'FAIL');document.querySelector('#lb').textContent=r.arm+' / Native '+(r.native_ok?'PASS':'FAIL');}
cs.onchange=load;ls.onchange=load;rs.onchange=load;load();
document.querySelector('#play').onclick=()=>{if(a.paused){b.currentTime=a.currentTime;Promise.all([a.play(),b.play()]).catch(()=>{});}else{a.pause();b.pause();}};
let syncing=false;
for(const [x,y] of [[a,b],[b,a]]){x.addEventListener('seeking',()=>{if(!syncing&&Math.abs(x.currentTime-y.currentTime)>.04){syncing=true;y.currentTime=x.currentTime;setTimeout(()=>syncing=false,80);}});x.addEventListener('pause',()=>{if(!syncing)y.pause();});}
a.addEventListener('play',()=>{if(b.paused){b.currentTime=a.currentTime;b.play().catch(()=>{});}});b.addEventListener('play',()=>{if(a.paused){a.currentTime=b.currentTime;a.play().catch(()=>{});}});
function tick(){if(!a.paused&&!b.paused&&Math.abs(a.currentTime-b.currentTime)>.08)b.currentTime=a.currentTime;requestAnimationFrame(tick);}tick();
document.querySelector('#zoom').oninput=e=>{for(const x of [a,b]){x.style.width=(600*Number(e.target.value))+'px';x.parentElement.scrollLeft=(x.parentElement.scrollWidth-x.parentElement.clientWidth)/2;}};
</script></html>"""
    (out/"index.html").write_text(page.replace("PAYLOAD",json.dumps(payload)),encoding="utf8")
    write(out/"protocol.json",dict(code_commit=commit(),records=media,selection="First preregistered Train probe and exposed DEV; both observation layouts",human_reviewed=False,operational_qualified=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(a.root,a.out)
