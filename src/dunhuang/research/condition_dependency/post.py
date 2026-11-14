"""Saved-artifact visualization only: fixed world camera, no smoothing or new sampling."""
import argparse,html,subprocess,sys
from pathlib import Path
import numpy as np
import soundfile as sf
from PIL import Image,ImageDraw,ImageFont
from dunhuang.io import checked,write,pinned
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.primitive_reperformance.render import PARENTS
from dunhuang.research.boundary_context.common import Data

def audio_for(wave,sr,start,frames,out):
    # Diagnostic media, not the independently sealed blind-review PCM24 protocol.
    first=round(start*sr/30);count=(frames*sr+29)//30
    clip=wave[first:first+count].copy()
    if len(clip)<count:clip=np.pad(clip,((0,count-len(clip)),(0,0)))
    sf.write(out,clip,sr,subtype="FLOAT")
    restored,rate=sf.read(out,dtype="float32",always_2d=True)
    if rate!=sr or not np.array_equal(clip,restored):raise ValueError("Float WAV diagnostic sample roundtrip differs")

def movie(arrays,labels,audio,file,title):
    width=400*len(arrays);height=540;N=max(map(len,arrays));font=ImageFont.load_default()
    points=np.concatenate([a.reshape(-1,3) for a in arrays]);project2=lambda a:np.stack([.9*a[...,0]-.44*a[...,2],-a[...,1]+.18*a[...,2]],-1)
    xy=project2(points);center=(xy.max(0)+xy.min(0))/2
    scale=min(350/max(float(np.ptp(xy[:,0])),1e-6),360/max(float(np.ptp(xy[:,1])),1e-6),200)
    floor=float(np.percentile(arrays[0][:,[7,8,10,11],1],5))
    low=points.min(0);high=points.max(0)
    cmd=["ffmpeg","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s",str(width)+"x"+str(height),"-r","30","-i","-","-i",str(audio),
        "-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart","-preset","veryfast","-crf","22","-c:a","aac","-shortest","-threads","2",str(file)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for t in range(N):
            im=Image.new("RGB",(width,height),"#f4f6f9");d=ImageDraw.Draw(im)
            for col,(a,label) in enumerate(zip(arrays,labels)):
                def screen(x):
                    q=project2(np.asarray(x));return (int(col*400+200+scale*(q[0]-center[0])),int(285+scale*(q[1]-center[1])))
                for x in np.arange(np.floor(low[0]*2)/2,high[0]+.5,.5):
                    d.line([screen([x,floor,low[2]-.25]),screen([x,floor,high[2]+.25])],fill="#d9dee5")
                for z in np.arange(np.floor(low[2]*2)/2,high[2]+.5,.5):
                    d.line([screen([low[0]-.25,floor,z]),screen([high[0]+.25,floor,z])],fill="#d9dee5")
                k=min(t,len(a)-1);pose=a[k];pts=[screen(v) for v in pose]
                for j in range(1,22):d.line([pts[PARENTS[j]],pts[j]],fill="#235e95" if j not in (20,21) else "#a5552a",width=3)
                for j in (7,8,10,11):
                    trail=[screen(v) for v in a[max(0,k-8):k+1,j]]
                    if len(trail)>1:d.line(trail,fill="#c75a48",width=2)
                for x,y in pts[:22]:d.ellipse((x-2,y-2,x+2,y+2),fill="#24344b")
                d.text((col*400+12,43),label,font=font,fill="#18304c")
                d.text((col*400+12,62),f"frame {k+1}/{len(a)} / "+("EDIT [60,90)" if 60<=t<90 else "OUTSIDE EDIT"),font=font,fill="#a5552a")
            d.text((12,12),title,font=font,fill="#18304c")
            d.text((12,498),"World Root retained. Shared fixed camera. 0.5m grid; floor is first-panel foot-height proxy.",font=font,fill="#18304c")
            d.text((12,518),"No smoothing. Foot trails: last 9 frames. Not a human style review; fingers unavailable.",font=font,fill="#18304c")
            proc.stdin.write(np.asarray(im).tobytes())
    finally:proc.stdin.close()
    if proc.wait():raise RuntimeError("World comparison encoding failed")
    return N

def run(training,out):
    training=Path(training);out=fresh(out);p=read(training/"generation/protocol.json");assets=read("inputs/clock_assets_v2.json")
    sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    from motion.geometry import fk_24_np
    data=Data(np.load(checked(assets["scaler"])));names=list(dict.fromkeys(b["case"] for b in p["bindings"]));videos=[]
    for name in names:
        bindings=[b for b in p["bindings"] if b["case"]==name];target=bindings[0]["target"]
        old=data.native(target) if name.startswith("val_") else np.load(checked(target["native"]))[:150]
        arrays=[fk_24_np(old)]+[np.load(checked(next(b for b in bindings if b["arm"]==arm)["fk"]))[:150] for arm in ("zero","C0","C1")]
        audio=out/(name+".wav");wave,sr=sf.read(checked(target["audio"]),dtype="float32",always_2d=True)
        start=target["start_frame"] if name.startswith("val_") else 0
        audio_for(wave,sr,start,150,audio);file=out/(name+".mp4")
        frames=movie(arrays,["PAIRED TARGET" if name.startswith("val_") else "FROZEN R8 CONTEXT","ZERO / FROZEN EDGE","C0 / REAL TARGET LOSS","C1 / + ROOT RESPONSE"],audio,file,"R10 actual window / "+name)
        videos.append(dict(name=name,frames=frames,video=ref(file),audio_clip=ref(audio),audio_sample_roundtrip=True,bindings=bindings,world_root_preserved=True))
    intro="""<h1>R10 单窗实际生成：固定世界坐标对照</h1><p>全部8个注册案例，不按成绩筛选。每行从左到右：真实配对AIST目标或冻结R8上下文、无Adapter、C0、C1。仅[60,90)中间30帧被替换，其他帧逐位保留。红色足迹是近9帧轨迹；不做平滑。诊断音频切片以FLOAT WAV保留原始解码采样，视频AAC音轨为有损呈现，不作为盲审封存证据。固定相机保留Root世界移动，网格0.5米，地面是首栏足高代理，并非人工地面真值。</p><p>C0/C1是双侧上下文条件实验，不能据此声称已实现完整ChangE来源演绎或敦煌舞风格。人类文化真值未知；来源与完整Native必须分别查看，不能凭姿态好看代替足滑验收。</p>"""
    cards=[]
    for v in videos:
        cards.append("<section><h2>"+html.escape(v["name"])+"</h2><video controls preload=metadata src="+html.escape(Path(v["video"]["path"]).name)+"></video><p>")
        for b in v["bindings"]:
            cards.append('<a href="../generation/'+html.escape(Path(b["native"]["path"]).parent.name)+'/edited_physical.json">'+html.escape(b["arm"])+' Native</a> ')
        cards.append("</p></section>")
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>R10单窗生成</title><style>body{font:16px sans-serif;background:#f4f6f9;max-width:1600px;margin:24px auto}p{line-height:1.65}section{background:white;padding:16px;margin:16px 0}video{width:100%}</style>'+intro+"".join(cards),encoding="utf8")
    checks=[]
    for v in videos:
        file=checked(v["video"]);probe=__import__("json").loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames,width,height,r_frame_rate","-of","json",str(file)],text=True))
        stream=probe["streams"][0];assert int(stream["nb_read_frames"])==150 and stream["r_frame_rate"]=="30/1"
        checks.append(dict(name=v["name"],video=v["video"],stream=stream))
    write(out/"protocol.json",dict(code_commit=commit(),training=ref(training/"generation/protocol.json"),videos=videos,neural_generations=0,no_smoothing=True,world_camera=True))
    write(out/"media_checks.json",checks)
    print("R10_MEDIA",len(videos),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--training",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(a.training,a.out)
