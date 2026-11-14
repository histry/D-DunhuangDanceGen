"""Train and exposed-DEV seed-matched videos, no smoothing or frame-wise recentering."""
import argparse,math,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.temporal_r20.common import parent_config,older_config,old_population

def run(root,out):
    root=Path(root);out=fresh(out);cfg=read(root/"interface/protocol.json")["config"]
    pre,rows,ev=old_population(older_config(parent_config(cfg)));native_environment(pre["assets"])
    from motion.geometry import fk_24_np
    from motion_geometry.smpl24 import PARENTS
    mapping={(b["case"],b["seed"],b["arm"]):root/"final_native"/folder for folder,b in read(root/"final_native/protocol.json")["saved"].items()}
    by={r["case"]:r for r in rows};arms=["S0","S1","A0","A1","A2"]+(["A3"] if read(root/"cache_gate.json")["cache_admitted"] else [])
    width=(len(arms)+1)*300;font=ImageFont.load_default();c,s=math.cos(-.45),math.sin(-.45);records=[]
    def project(x):return np.stack((c*x[...,0]+s*x[...,2],x[...,1]-.18*(-s*x[...,0]+c*x[...,2])),-1)
    for case in cfg["train_cases"]+cfg["dev_cases"]:
        role="TRAIN" if case in cfg["train_cases"] else "EXPOSED DEV"
        row=by[case];z=np.load(checked(row["arrays"]))
        for seed in cfg["eval_seeds"]:
            tracks=[fk_24_np(z["native"])]+[np.load(mapping[case,seed,a]/"fk.npy") for a in arms]
            xy=np.stack([project(x) for x in tracks]);lo=xy.min((0,1,2));hi=xy.max((0,1,2));center=(lo+hi)/2
            scale=min(260/max(float(hi[0]-lo[0]),1e-6),450/max(float(hi[1]-lo[1]),1e-6))
            path=out/f"case{case}_seed{seed}.mp4";audio=checked(row["audio225"])
            proc=subprocess.Popen(["ffmpeg","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x600","-r","30","-i","-",
                "-i",str(audio),"-map","0:v:0","-map","1:a:0","-c:v","libx264","-pix_fmt","yuv420p","-preset","veryfast","-crf","20",
                "-threads","2","-c:a","aac","-shortest",str(path)],stdin=subprocess.PIPE)
            try:
                for t in range(225):
                    image=Image.new("RGB",(width,600),"#f2f4f7");draw=ImageDraw.Draw(image)
                    for k,label in enumerate(["Paired AIST reference"]+arms):
                        off=k*300;draw.text((off+10,12),label,fill="#173249",font=font)
                        draw.text((off+10,32),f"{role}{case} seed{seed} | {t+1}/225",fill="#40556c",font=font)
                        pts=[(round(off+150+(p[0]-center[0])*scale),round(315-(p[1]-center[1])*scale)) for p in xy[k,t]]
                        for j in range(1,24):draw.line([pts[int(PARENTS[j])],pts[j]],fill="#285e99" if j%2 else "#9c5133",width=3)
                        for a,b in pts:draw.ellipse((a-2,b-2,a+2,b+2),fill="#22394c")
                        draw.text((off+10,553),"MECHANISM / UNQUALIFIED",fill="#415b6b",font=font)
                        draw.text((off+10,575),"World Root / no smoothing",fill="#415b6b",font=font)
                    proc.stdin.write(np.asarray(image).tobytes())
            finally:proc.stdin.close()
            assert proc.wait()==0
            frames=int(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","csv=p=0",str(path)],text=True).strip())
            assert frames==225;subprocess.run(["ffmpeg","-v","error","-i",str(path),"-f","null","-"],check=True)
            records.append(dict(case=case,role=role,seed=seed,video=ref(path),audio=row["audio225"],frames=225,full_decode_passed=True))
            print("R20_VIDEO",path.name,flush=True)
    cards="".join('<h2>'+r["role"]+' '+str(r["case"])+' seed'+str(r["seed"])+'</h2><video controls preload="metadata" src="'+Path(r["video"]["path"]).name+'"></video>' for r in records)
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>R20双端结构与时序</title><style>body{max-width:2100px;margin:24px auto;font:16px sans-serif}video{width:100%}</style><h1>R20：S0双150 / S1真225 / A0额外训练 / A1时刻支持 / A2全身时序 / A3缓存若准入</h1><p>AIST已知双端补全；两段Train、两段历史曝光DEV，两个固定种子。DEV未参与结构或缓存准入。并非无未来Root的整曲敦煌舞；不代表Native合格。真实音乐、固定世界坐标，无平滑。</p>'+cards,encoding="utf-8")
    write(out/"protocol.json",dict(code_commit=commit(),records=records,human_reviewed=False,operational_qualified=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(a.root,a.out)
