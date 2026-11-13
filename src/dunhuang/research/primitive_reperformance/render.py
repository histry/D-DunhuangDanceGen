"""Three-way skeleton evidence; 22 observed joints, no invented fingers or Root truth."""
import argparse,math,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from PIL import Image,ImageDraw,ImageFont
from dunhuang.io import checked,write,pinned
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.source_expression.review import audio_clip
from .diagnosis import canonical
PARENTS=(-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21)
def movie(arrays,audio,file,title,wrist_glyphs=False):
    N=max(len(a) for a in arrays);font=ImageFont.load_default();width,height=1200,520
    allpoints=np.concatenate([a[:,:24 if wrist_glyphs else 22].reshape(-1,3) for a in arrays])
    px=.9*allpoints[:,0]-.44*allpoints[:,2];py=-allpoints[:,1]+(20/115)*allpoints[:,2]
    scale=min(360/max(float(np.ptp(px)),1e-6),350/max(float(np.ptp(py)),1e-6),220.)
    center_x=float((px.max()+px.min())/2);center_y=float((py.max()+py.min())/2)
    cmd=["ffmpeg","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","1200x520","-r","30","-i","-","-i",str(audio),"-c:v","libx264","-preset","veryfast","-crf","22","-c:a","aac","-shortest","-threads","2",str(file)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for t in range(N):
            im=Image.new("RGB",(width,height),"#f3f5f8");draw=ImageDraw.Draw(im)
            for panel,(label,x) in enumerate(zip(("RAW FITTED CHANGE","LEGAL E1-W REFERENCE","EDGE OUTPUT"),arrays)):
                points=x[min(t,len(x)-1)];origin=200+400*panel
                def project(p):return (round(origin+scale*(.9*p[0]-.44*p[2]-center_x)),round(275+scale*(-p[1]+(20/115)*p[2]-center_y)))
                pts=[project(p) for p in points]
                for j in range(1,22):draw.line([pts[PARENTS[j]],pts[j]],fill="#ab562f" if j in (20,21) else "#265f95",width=3)
                if wrist_glyphs:
                    for wrist,tip in ((20,22),(21,23)):
                        start=np.asarray(pts[wrist],float);end=np.asarray(pts[tip],float)
                        for u,v in ((0,.25),(.4,.65),(.8,1)):
                            draw.line([tuple(start+u*(end-start)),tuple(start+v*(end-start))],fill="#ab562f",width=3)
                        a,b=pts[tip];draw.ellipse((a-3,b-3,a+3,b+3),outline="#ab562f",width=1)
                for a,b in pts[:22]:draw.ellipse((a-2,b-2,a+2,b+2),fill="#24344b")
                draw.text((panel*400+12,45),label,font=font,fill="#182b43")
                draw.text((panel*400+12,65),f"frame {min(t+1,len(x))}/{len(x)}"+(" END HOLD" if t>=len(x) else ""),font=font,fill="#182b43")
            draw.text((12,12),title,font=font,fill="#182b43")
            draw.text((12,480),"Pelvis-relative, initial heading canonicalized; 22 fitted joints. Unknown world Root/fingers. Orange: wrists.",font=font,fill="#182b43")
            draw.text((12,500),"Raw source uses its original 30FPS clock; ended panels hold their last frame only for display.",font=font,fill="#182b43")
            process.stdin.write(np.asarray(im).tobytes())
    finally:process.stdin.close()
    if process.wait():raise RuntimeError("Three-way video encoder failed")
    return N
def audio_for(wave,sr,start,frames,out):
    a=round(start*sr/30);n=(frames*sr+29)//30;x=wave[a:a+n]
    if len(x)<n:x=np.pad(x,((0,n-len(x)),(0,0)))
    audio_clip(out,x.mean(1)[:,None],sr)
def run(cfg,diagnosis,prepared,single,out,wrist_glyphs=False):
    import sys
    out=fresh(out);assets=read(cfg["assets"]);sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    from motion.geometry import fk_24_np
    prior=Path(cfg["prior_song"]);inventory=read(checked(read(prior/"music/protocol.json")["inventory"]));videos=[]
    d=read(Path(diagnosis)/"result.json")
    for row in d["records"]:
        values=np.load(checked(row["arrays"]));arrays=[values[k] for k in ("raw","retimed","generated")];N=max(map(len,arrays))
        music=inventory[row["music_case"]];wave,sr=sf.read(checked(music["audio"]),dtype="float32",always_2d=True)
        name="old_"+str(row["case"]);audio=out/(name+".wav");audio_for(wave,sr,row["music_interval"][0],N,audio);file=out/(name+".mp4")
        frames=movie(arrays,audio,file,"R8-S0 / original R7 / item "+str(row["case"]),wrist_glyphs);videos.append(dict(kind="R8-S0",name=name,frames=frames,video=ref(file),evidence=row))
    p=read(Path(prepared)/"protocol.json");pairs=read(checked(p["pairs"]));music=read(checked(p["music"]));result=read(Path(single)/"result.json");bindings=read(Path(single)/"protocol.json")["bindings"]
    for pair in pairs:
        for label in ("A","B"):
            plan=read(checked(pair["plans"][label]));e=plan["events"][0];source=np.load(checked(e["reference"]));lo,hi=e["source_interval"];raw=source[lo:hi]
            from dunhuang.generation.soft_reference import resample_native
            from dunhuang.planning.time_map import source_positions
            aligned=resample_native(source,source_positions(e,np.arange(pair["frames"])))
            for song in music:
                name=str(pair["pair"])+"_"+str(song["case"])+"_S3_"+label
                binding=next(b for b in bindings if b["name"]==name);generated=np.load(checked(binding["native"]))
                arrays=[canonical(fk_24_np(v),v) for v in (raw,aligned,generated)];N=max(map(len,arrays))
                wave,sr=sf.read(checked(song["audio"]),dtype="float32",always_2d=True);audio=out/(name+".wav");audio_for(wave,sr,0,N,audio);file=out/(name+".mp4")
                frames=movie(arrays,audio,file,"R8-S3 / pair "+str(pair["pair"])+" / source "+label+" / "+song["name"],wrist_glyphs)
                measure=next(r for r in result["records"] if r["pair"]==pair["pair"] and r["music"]==song["case"] and r["arm"]=="S3" and r["source_label"]==label)
                videos.append(dict(kind="R8-S3",name=name,frames=frames,video=ref(file),native_ok=measure["native_ok"],source=measure["source"]))
    cards=[]
    for v in videos:
        caption=v["kind"]+" / "+v["name"]
        if "native_ok" in v:caption+=" / Native="+str(v["native_ok"])+" / source="+format(v["source"]["mean_source_rad"],".3f")+" rad"
        cards.append('<section><h2>'+caption+'</h2><video controls preload="metadata" src="'+Path(v["video"]["path"]).name+'"></video></section>')
    text='<h1>ChangE → E1-W → EDGE 三方诊断</h1><p>前组展示旧R7；后组展示新S3。原始拟合骨架、重定时参考、生成结果使用相同骨架和相机。Root为相对视图，手指没有观测；自动阶段与风格真实性尚未获得人工真值。不同长度的面板结束后保持最后一帧，原始动作不改速。</p>'
    if wrist_glyphs:text+='<p>橙色虚线仅表示已观测腕旋转与固定SMPL偏移给出的手部方向；末端关节旋转和手指仍未观测。</p>'
    text+='<p>单事件联合门：'+str(result["single_event_gate_passed"])+'；新整曲必须同时通过来源与Bridge检查。</p>'
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>R8三方诊断</title><style>body{max-width:1200px;margin:24px auto;padding:16px;background:#f3f5f8;font:16px sans-serif;color:#182b43}video{width:100%}section{background:white;padding:16px;margin:20px 0}p{line-height:1.6}</style>'+text+"".join(cards),encoding="utf-8")
    write(out/"protocol.json",dict(code_commit=commit(),videos=videos,frame_rate=30,wrist_direction_glyphs=wrist_glyphs,finger_geometry_observed=False,camera="one shared fixed fit to all three complete trajectories per clip",neural_generation_calls=0,human_reviewed=False,operational_qualified=False))
    print("R8_THREE_WAY_VIDEOS",len(videos),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--diagnosis",required=True);p.add_argument("--prepared",required=True);p.add_argument("--single",required=True);p.add_argument("--out",required=True);p.add_argument("--wrist-glyphs",action="store_true");a=p.parse_args();run(read(a.config),a.diagnosis,a.prepared,a.single,a.out,a.wrist_glyphs)
