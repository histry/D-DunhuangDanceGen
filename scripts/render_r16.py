"""Fixed-camera source/C0/C1 diagnostics; no smoothing and no motion edits."""
import argparse,subprocess,sys,math,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.state_bridge.common import native_environment

def run(root,out):
    out=fresh(out);root=Path(root);pre=read(root/"preflight/protocol.json");native_environment(pre["assets"])
    from motion.geometry import fk_24_np
    from motion_geometry.smpl24 import PARENTS as native_parents
    parents=list(map(int,native_parents));records=[]
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):raise ValueError("ffmpeg/ffprobe required")
    yaw=-.45;cs,sn=math.cos(yaw),math.sin(yaw);font=ImageFont.load_default()
    for case,b in enumerate(pre["pairs"]):
        z=np.load(checked(b["arrays"]))
        arrays=[fk_24_np(z["native"][:150]),np.load(root/"native"/(str(case)+"_C0")/"fk.npy"),
            np.load(root/"native"/(str(case)+"_C1_correct")/"fk.npy")]
        def xy(a):return np.stack((cs*a[...,0]+sn*a[...,2],a[...,1]-.18*(-sn*a[...,0]+cs*a[...,2])),-1)
        points=np.stack([xy(a) for a in arrays]);lo=points.min((0,1,2));hi=points.max((0,1,2));span=hi-lo
        scale=min(370/max(float(span[0]),1e-6),470/max(float(span[1]),1e-6));center=(lo+hi)/2
        audio=checked(read(checked(b["producer"][0]))["crop"]);file=out/("case_"+str(case)+".mp4")
        proc=subprocess.Popen(["ffmpeg","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","1260x600","-r","30","-i","-","-i",str(audio),
            "-map","0:v:0","-map","1:a:0","-c:v","libx264","-pix_fmt","yuv420p","-preset","veryfast","-crf","20","-threads","2","-c:a","aac","-shortest",str(file)],stdin=subprocess.PIPE)
        try:
            for t in range(150):
                image=Image.new("RGB",(1260,600),"#f2f4f7");draw=ImageDraw.Draw(image)
                for k,label in enumerate(("Real paired source","C0: old 30-frame graft","C1: full-window generated")):
                    offset=k*420
                    draw.text((offset+12,14),label,fill="#172e48",font=font)
                    draw.text((offset+12,34),f"frame {t+1}/150 | {t/30:.2f}s | UNQUALIFIED",fill="#3c5268",font=font)
                    def screen(p):
                        v=(p-center)*scale;return (round(offset+210+v[0]),round(315-v[1]))
                    coords=[screen(p) for p in points[k,t]]
                    for j in range(1,24):draw.line([coords[parents[j]],coords[j]],fill="#285d99" if j%2 else "#9c4f31",width=3)
                    for p in coords:draw.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill="#20394e")
                    for foot in (7,8,10,11):
                        track=[screen(p) for p in points[k,max(0,t-12):t+1,foot]]
                        if len(track)>1:draw.line(track,fill="#9bb7bd",width=1)
                    draw.text((offset+12,575),"Fixed world camera; actual FK; no smoothing",fill="#40566b",font=font)
                proc.stdin.write(np.asarray(image).tobytes())
        finally:proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError("Video encode failed")
        count=int(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","csv=p=0",str(file)],text=True).strip())
        assert count==150
        subprocess.run(["ffmpeg","-v","error","-i",str(file),"-f","null","-"],check=True)
        records.append(dict(case=case,video=ref(file),frames=count,audio=ref(audio),full_decode_passed=True,
            camera="Fixed per case, same bounds for all panels; real world FK, no root recentering, no smoothing",perceptual_qualification=False))
        print("R16_VIDEO",case,flush=True)
    cards="".join('<section><h2>Case '+str(r["case"])+'</h2><video controls preload="metadata" src="'+Path(r["video"]["path"]).name+'"></video></section>' for r in records)
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>R16 生成诊断</title><style>body{max-width:1280px;margin:24px auto;font:16px sans-serif}video{width:100%}</style><h1>R16：真实来源 / C0 / C1</h1><p>固定世界视角、同一真实音乐和时间轴；无平滑、无逐帧Root重定心。用于诊断，不代表Native或风格合格，不采集人工标签。</p>'+cards,encoding="utf-8")
    write(out/"protocol.json",dict(code_commit=commit(),records=records,optimizer_steps=0,human_reviewed=False,operational_qualified=False))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(a.root,a.out)
