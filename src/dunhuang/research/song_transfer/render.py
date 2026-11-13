"""Complete generated-motion diagnostic videos; not a perceptual study."""
import argparse,subprocess,math,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
PARENTS=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
def video(joints,audio,path,label,plane_y):
    if shutil.which("ffmpeg") is None:raise RuntimeError("ffmpeg required for full videos")
    width,height=960,640;scale=160.;yaw=-.45;cy=float(np.median(joints[:,0,1]));c,s=math.cos(yaw),math.sin(yaw)
    command=["ffmpeg","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s",str(width)+"x"+str(height),"-r","30","-i","-","-i",str(audio),
        "-map","0:v:0","-map","1:a:0","-c:v","libx264","-preset","veryfast","-crf","20","-c:a","aac","-shortest",str(path)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for t,points in enumerate(joints):
            image=Image.new("RGB",(width,height),"#f3f5f8");draw=ImageDraw.Draw(image)
            root=points[0].copy();center=np.array([root[0],cy,root[2]])
            def project(p):
                d=p-center;return (round(width/2+scale*(c*d[0]+s*d[2])),round(340-scale*d[1]+scale*.22*(-s*d[0]+c*d[2])))
            # Reference grid follows world coordinates, not the body. Plane is diagnostic only.
            for a in range(math.floor(root[0])-5,math.floor(root[0])+6):draw.line([project([a,plane_y,root[2]-5]),project([a,plane_y,root[2]+5])],fill="#d6dde7",width=1)
            for b in range(math.floor(root[2])-5,math.floor(root[2])+6):draw.line([project([root[0]-5,plane_y,b]),project([root[0]+5,plane_y,b])],fill="#d6dde7",width=1)
            coords=[project(p) for p in points]
            for j in range(1,24):draw.line([coords[PARENTS[j]],coords[j]],fill="#255f9b" if j%2 else "#9b4a25",width=4)
            for a,b in coords:draw.ellipse((a-3,b-3,a+3,b+3),fill="#24344b")
            draw.text((18,18),label,fill="#182b43");draw.text((18,40),f"frame {t+1}/{len(joints)}  {t/30:.2f} s",fill="#182b43")
            draw.text((18,height-36),"Root-follow camera; Native-estimated plane; no observed ground/contact truth.",fill="#40536c")
            process.stdin.write(np.asarray(image).tobytes())
    finally:process.stdin.close()
    if process.wait()!=0:raise RuntimeError("Full video encoder failed")
def run(generation,features,out):
    out=fresh(out);protocol=read(Path(generation)/"protocol.json");result=read(Path(generation)/"result.json");songs=read(Path(features)/"protocol.json")["songs"];records=[]
    for row,binding in zip(result["records"],protocol["bindings"]):
        case=row["case"];arm=row["arm"];joints=np.load(checked(binding["fk"]));audio=checked(songs[case]["audio"]);path=out/(str(case)+"_"+arm+".mp4")
        if len(joints)!=row["frames"]:raise ValueError("Video is not the full song")
        physical=read(checked(binding["physical"]));plane_y=physical["audit"]["floor_y_m"];name=Path(songs[case]["audio"]["path"]).name
        video(joints,audio,path,"UNQUALIFIED DEVELOPMENT / "+name+" / "+arm,plane_y)
        records.append(dict(case=case,arm=arm,name=name,frames=len(joints),video=ref(path),physical=binding["physical"],plane_y_m=plane_y,
            native_ok=row["native_ok"],jerk_p95_mps3=physical["audit"]["joint_jerk_mps3_p95"],foot_skate_p95_mps=physical["audit"]["foot_skate_mps_p95"]))
        print("R7_FULL_VIDEO",case,arm,len(joints),flush=True)
    def card(r):
        title=r["name"]+" / "+r["arm"]
        return '<section><h2>'+title+'</h2><p>Native: '+str(r["native_ok"])+'; Jerk P95: '+format(r["jerk_p95_mps3"],".2f")+' m/s³; foot skate P95: '+format(r["foot_skate_p95_mps"],".3f")+' m/s</p><video controls preload="metadata" src="'+Path(r["video"]["path"]).name+'"></video></section>'
    primary="".join(card(r) for r in records if r["arm"]=="temporal_A")
    controls="".join(card(r) for r in records if r["arm"]!="temporal_A")
    page='<p>三首完整长音乐的开发结果。主计划A使用既有Temporal关键姿态条件，学习式Adapter尚未接入；动作质量未获准。本页展示诊断视频，不采集盲审评分。参考平面来自Native估计，不是观测地面真值。</p><h1>主计划A：三首完整动作</h1>'+primary+'<h1>其余固定对照</h1>'+controls
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>整曲开发结果（未合格）</title><style>body{max-width:1000px;margin:24px auto;padding:0 18px;font:16px sans-serif;background:#f3f5f8;color:#182b43}section{background:white;padding:16px;margin:20px 0;border-radius:8px}video{width:100%;max-width:960px}p{line-height:1.6}</style>'+page,encoding="utf-8")
    write(out/"protocol.json",dict(code_commit=commit(),generation=ref(Path(generation)/"protocol.json"),records=records,full_song=True,plane_policy="saved Native floor estimate, not observed contact; no motion change or smoothing",human_reviewed=False,operational_qualified=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--generation",required=True);p.add_argument("--features",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(a.generation,a.features,a.out)
