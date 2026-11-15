"""Fixed world camera, identical scale per source/family; no motion modification."""
import argparse,math,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.local_r23.common import read,fresh,inputs,commit
from dunhuang.research.state_bridge.common import native_environment

def video(path,points,parents,label,ok,center,scale,audio=None):
    cmd=["ffmpeg","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","600x600","-r","30","-i","-"]
    if audio:cmd+=["-i",str(audio),"-map","0:v:0","-map","1:a:0"]
    cmd+=["-c:v","libx264","-pix_fmt","yuv420p","-preset","veryfast","-crf","20","-threads","2"]
    if audio:cmd+=["-c:a","aac","-shortest"]
    cmd+=[str(path)];proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for t,frame in enumerate(points):
            image=Image.new("RGB",(600,600),"#f2f4f7");draw=ImageDraw.Draw(image)
            draw.text((12,12),label,fill="#173249");draw.text((12,34),"Native "+("PASS" if ok else "FAIL")+" | AIST mechanism, not Dunhuang artwork",fill="#70402e")
            pts=[(round(300+(v[0]-center[0])*scale),round(310-(v[1]-center[1])*scale)) for v in frame]
            for j in range(1,24):draw.line([pts[parents[j]],pts[j]],fill="#285e99" if j%2 else "#9c5133",width=4)
            for x,y in pts:draw.ellipse((x-3,y-3,x+3,y+3),fill="#22394c")
            draw.text((12,553),f"{t+1}/225 | fixed world camera, common scale",fill="#415b6b")
            draw.text((12,576),"No smoothing / no frame recentering / unqualified research",fill="#415b6b")
            proc.stdin.write(np.asarray(image).tobytes())
    finally:proc.stdin.close()
    assert proc.wait()==0
    n=int(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","csv=p=0",str(path)],text=True).strip())
    assert n==225;subprocess.run(["ffmpeg","-v","error","-i",str(path),"-f","null","-"],check=True)

def run(cfg,study,out):
    study=Path(study);out=fresh(out);root,pc,pop=inputs(cfg);native_environment(read(root/"data/protocol.json")["assets"])
    from motion_geometry.smpl24 import PARENTS
    from motion.geometry import fk_24_np
    native=read(study/"native/result.json")["records"];rep=read(study/"represent/result.json");media=[]
    groups=[]
    for case in [0,24]:
        row=next(r for r in pop["records"] if r["case"]==case);source=np.load(checked(row["arrays"]))["native"]
        tracks=[("Source",fk_24_np(source),read(checked(row["source_physical"]))["absolute"]["ok"])]
        for arm in cfg["arms"]:
            for weight in cfg["guidance"]:
                r=next(r for r in native if (r["arm"],r["case"],r["variant"],r["cfg"])==(arm,case,0,weight))
                tracks.append((arm+"_cfg"+str(weight),np.load(study/"native"/r["array"])["fk"],r["native_ok"]))
        groups.append(("CFG_case"+str(case),tracks,checked(row["audio225"])))
    for case in [0,5]:
        rows=[r for r in rep["records"] if r["case"]==case]
        tracks=[(r["mode"],np.load(study/"represent"/r["array"])["fk"],r["native_ok"]) for r in rows]
        groups.append(("Representation_"+rows[0]["role"]+"_case"+str(case),tracks,None))
    c,s=math.cos(-.45),math.sin(-.45)
    def project(x):return np.stack((c*x[...,0]+s*x[...,2],x[...,1]-.18*(-s*x[...,0]+c*x[...,2])),-1)
    for group,tracks,audio in groups:
        xy=np.stack([project(points) for label,points,ok in tracks]);lo=xy.min((0,1,2));hi=xy.max((0,1,2));center=(lo+hi)/2
        scale=min(540/max(float(hi[0]-lo[0]),1e-6),440/max(float(hi[1]-lo[1]),1e-6))
        for label,points,ok in tracks:
            path=out/(group+"_"+label+".mp4");video(path,project(points),PARENTS,group+" | "+label,ok,center,scale,audio)
            media.append(dict(group=group,label=label,file=path.name,video=ref(path),native_ok=ok,frames=225,full_decode_passed=True))
        print("R23_VIDEO_GROUP",group,flush=True)
    page="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>R23 局部生成研究对照</title>
<style>body{margin:24px;font:16px system-ui;background:#f2f4f7;color:#173249}select,button{padding:8px;margin:6px}.pair{display:flex;gap:20px}video{width:48%;background:white}</style>
<h1>R23：固定引导与完整连续表示</h1><p>CFG 对照为配对 AIST 补全；表示对照为完整目标拟合上限，不是条件生成。均不是敦煌舞成品。原 Native 不变；相同来源使用相同世界视角及比例，不平滑、不逐帧居中。表示视频无音乐。</p>
<select id="group"></select><select id="left"></select><select id="right"></select><button id="play">同步播放／暂停</button><p id="labels"></p>
<div class="pair"><video id="a" controls preload="metadata"></video><video id="b" controls muted preload="metadata"></video></div>
<script>const rows=PAYLOAD;const group=document.querySelector('#group'),left=document.querySelector('#left'),right=document.querySelector('#right'),a=document.querySelector('#a'),b=document.querySelector('#b');
for(const g of new Set(rows.map(r=>r.group)))group.add(new Option(g,g));
function load(){const rs=rows.filter(r=>r.group===group.value),x=rs.find(r=>r.label===left.value),y=rs.find(r=>r.label===right.value);a.pause();b.pause();a.src=x.file;b.src=y.file;document.querySelector('#labels').textContent=x.label+' Native '+x.native_ok+' / '+y.label+' Native '+y.native_ok;}
function populate(){left.innerHTML='';right.innerHTML='';const rs=rows.filter(r=>r.group===group.value);for(const r of rs){left.add(new Option(r.label,r.label));right.add(new Option(r.label,r.label));}right.selectedIndex=rs.length-1;load();}
group.onchange=populate;left.onchange=load;right.onchange=load;populate();document.querySelector('#play').onclick=()=>{if(a.paused){b.currentTime=a.currentTime;Promise.all([a.play(),b.play()]).catch(()=>{});}else{a.pause();b.pause();}};
function tick(){if(!a.paused&&!b.paused&&Math.abs(a.currentTime-b.currentTime)>.08)b.currentTime=a.currentTime;requestAnimationFrame(tick);}tick();</script></html>"""
    (out/"index.html").write_text(page.replace("PAYLOAD",json.dumps(media)),encoding="utf8")
    write(out/"protocol.json",dict(code_commit=commit(),records=media,live_controls_verified=False,fixed_world_camera=True,motion_smoothing=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--study",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    run(read(a.config),a.study,a.out)
