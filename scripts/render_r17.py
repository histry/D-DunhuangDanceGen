"""Same-clock Source/A/B/C videos with a fixed world camera, never smoothed."""
import argparse,math,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.state_bridge.common import native_environment

def run(root,out):
    root=Path(root);out=fresh(out);pre=read(root/"data/protocol.json");native_environment(pre["assets"])
    from motion.geometry import fk_24_np
    from motion_geometry.smpl24 import PARENTS
    rows={r["case"]:r for r in read(root/"data/result.json")["records"] if r["role"]=="eval"}
    native=read(root/"native/result.json");mapping={}
    for folder,entry in read(root/"native/protocol.json")["saved"].items():
        if entry["kind"]!="rollout":continue
        # Final Native directories follow the frozen audit40, then rollout ordering.
        ordinal=int(Path(folder).name.rsplit("_",1)[1])-40
        r=read(root/"rollout/result.json")["bindings"][ordinal];mapping[r["case"],r["arm"],r["task"]]=Path(folder)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):raise ValueError("Video encoders unavailable")
    c,s=math.cos(-.45),math.sin(-.45);font=ImageFont.load_default();records=[]
    def project(x):return np.stack((c*x[...,0]+s*x[...,2],x[...,1]-.18*(-s*x[...,0]+c*x[...,2])),-1)
    for task in ("free","inbetween","progressive"):
        for case,row in rows.items():
            z=np.load(checked(row["arrays"]));arrays=[fk_24_np(z["native"])]+[np.load(mapping[case,arm,task]/"fk.npy") for arm in ("A_numeric","B_cross","C_self")]
            xy=np.stack([project(x) for x in arrays]);lo=xy.min((0,1,2));hi=xy.max((0,1,2));span=hi-lo
            scale=min(270/max(float(span[0]),1e-6),470/max(float(span[1]),1e-6));center=(lo+hi)/2
            path=out/(str(case)+"_"+task+".mp4");audio=checked(row["audio225"])
            proc=subprocess.Popen(["ffmpeg","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","1280x600","-r","30","-i","-","-i",str(audio),
                "-map","0:v:0","-map","1:a:0","-c:v","libx264","-pix_fmt","yuv420p","-preset","veryfast","-crf","20","-threads","2","-c:a","aac","-shortest",str(path)],stdin=subprocess.PIPE)
            try:
                for t in range(225):
                    image=Image.new("RGB",(1280,600),"#f2f4f7");draw=ImageDraw.Draw(image)
                    for k,label in enumerate(("Paired reference","A numeric consensus","B learned cross-state","C matched self-state")):
                        off=k*320
                        draw.text((off+10,12),label,fill="#173249",font=font)
                        draw.text((off+10,32),f"AIST++ {task} | {t+1}/225 | {t/30:.2f}s",fill="#40556c",font=font)
                        def screen(p):
                            v=(p-center)*scale;return (round(off+160+v[0]),round(315-v[1]))
                        points=[screen(p) for p in xy[k,t]]
                        for j in range(1,24):draw.line([points[int(PARENTS[j])],points[j]],fill="#285e99" if j%2 else "#9c5133",width=3)
                        for a,b in points:draw.ellipse((a-2,b-2,a+2,b+2),fill="#22394c")
                        for foot in (7,8,10,11):
                            trail=[screen(p) for p in xy[k,max(0,t-12):t+1,foot]]
                            if len(trail)>1:draw.line(trail,fill="#a9bbc5",width=1)
                        status="GENERATED PREFIX" if task=="progressive" and t<75 and k else "RESEARCH / UNQUALIFIED"
                        draw.text((off+10,553),status,fill="#415b6b",font=font)
                        draw.text((off+10,575),"Fixed world / no smoothing",fill="#415b6b",font=font)
                    proc.stdin.write(np.asarray(image).tobytes())
            finally:proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError("Video encoding failed")
            frames=int(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","csv=p=0",str(path)],text=True).strip())
            assert frames==225;subprocess.run(["ffmpeg","-v","error","-i",str(path),"-f","null","-"],check=True)
            records.append(dict(case=case,task=task,video=ref(path),audio=row["audio225"],frames=225,full_decode_passed=True,operational_qualified=False))
            print("R17_VIDEO",task,case,flush=True)
    cards="".join('<section><h2>'+r["task"]+' / Case '+str(r["case"])+'</h2><video controls preload="metadata" src="'+Path(r["video"]["path"]).name+'"></video></section>' for r in records)
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>R17 AIST++机制诊断</title><style>body{max-width:1300px;margin:24px auto;font:16px sans-serif}video{width:100%}</style><h1>R17：真实参考 / 数值共享 / 跨窗口学习 / 同预算窗口内学习</h1><p>同一225帧音乐和世界坐标，固定相机、无平滑。FREE没有未来Root真值输入；INBETWEEN有真实边界；PROGRESSIVE共用A实际生成的75帧历史。此处是AIST++机制诊断，不是敦煌舞风格成果，也不代表Native合格。</p>'+cards,encoding="utf-8")
    write(out/"protocol.json",dict(code_commit=commit(),records=records,human_reviewed=False,perceptual_qualification=False,operational_qualified=False))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(a.root,a.out)
