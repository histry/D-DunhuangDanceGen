"""Fixed-world 225-frame reference/D0/D1/D2 videos; no smoothing or recentering."""
import argparse,math,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.state_bridge.common import native_environment

def run(root,out):
    root=Path(root);out=fresh(out);cfg=read(root/"p0/protocol.json")["config"]
    pre=read(Path(cfg["parent"])/"data/protocol.json");native_environment(pre["assets"])
    from motion.geometry import fk_24_np
    from motion_geometry.smpl24 import PARENTS
    rows={(r["role"],r["case"]):r for r in read(Path(cfg["parent"])/"data/result.json")["records"]}
    mapping={}
    for folder,b in read(root/"native/protocol.json")["saved"].items():
        if b["control"]!="correct":continue
        phase=folder.rsplit("_",1)[0];mapping[phase,b["case"],b["task"],b["arm"]]=root/"native"/folder
    records=[];c,s=math.cos(-.45),math.sin(-.45);font=ImageFont.load_default()
    def project(x):return np.stack((c*x[...,0]+s*x[...,2],x[...,1]-.18*(-s*x[...,0]+c*x[...,2])),-1)
    for phase in ["tiny"]+(["main"] if (root/"main_rollout").exists() else []):
        cases=cfg["tiny_cases"] if phase=="tiny" else list(range(4))
        for case in cases:
            row=rows["fit" if phase=="tiny" else "eval",case];z=np.load(checked(row["arrays"]))
            for task in cfg["tasks"]:
                tracks=[fk_24_np(z["native"])]+[np.load(mapping[phase,case,task,arm]/"fk.npy") for arm in ("D0","D1","D2")]
                xy=np.stack([project(t) for t in tracks]);lo=xy.min((0,1,2));hi=xy.max((0,1,2));center=(lo+hi)/2
                scale=min(270/max(float(hi[0]-lo[0]),1e-6),460/max(float(hi[1]-lo[1]),1e-6))
                path=out/f"{phase}_{case}_{task}.mp4";audio=checked(row["audio225"])
                proc=subprocess.Popen(["ffmpeg","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","1280x600","-r","30","-i","-",
                    "-i",str(audio),"-map","0:v:0","-map","1:a:0","-c:v","libx264","-pix_fmt","yuv420p","-preset","veryfast",
                    "-crf","20","-threads","2","-c:a","aac","-shortest",str(path)],stdin=subprocess.PIPE)
                try:
                    for t in range(225):
                        image=Image.new("RGB",(1280,600),"#f2f4f7");draw=ImageDraw.Draw(image)
                        for k,label in enumerate(("Paired AIST reference","D0 frozen numeric","D1 adapted self","D2 adapted unique")):
                            off=k*320;draw.text((off+10,12),label,fill="#173249",font=font)
                            draw.text((off+10,32),f"{phase} {case} {task} | {t+1}/225",fill="#40556c",font=font)
                            def screen(p):
                                v=(p-center)*scale;return round(off+160+v[0]),round(315-v[1])
                            points=[screen(p) for p in xy[k,t]]
                            for j in range(1,24):draw.line([points[int(PARENTS[j])],points[j]],fill="#285e99" if j%2 else "#9c5133",width=3)
                            for a,b in points:draw.ellipse((a-2,b-2,a+2,b+2),fill="#22394c")
                            draw.text((off+10,553),"TRAIN OVERFIT" if phase=="tiny" else "EXPOSED EVAL",fill="#415b6b",font=font)
                            draw.text((off+10,575),"World / no smoothing / unqualified",fill="#415b6b",font=font)
                        proc.stdin.write(np.asarray(image).tobytes())
                finally:proc.stdin.close()
                if proc.wait()!=0:raise RuntimeError("Video encoding failed")
                frames=int(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries",
                    "stream=nb_read_frames","-of","csv=p=0",str(path)],text=True).strip())
                assert frames==225;subprocess.run(["ffmpeg","-v","error","-i",str(path),"-f","null","-"],check=True)
                records.append(dict(phase=phase,case=case,task=task,video=ref(path),frames=frames,full_decode_passed=True))
                print("R18_VIDEO",path.name,flush=True)
    cards="".join('<h2>'+r["phase"]+' '+str(r["case"])+' '+r["task"]+'</h2><video controls preload="metadata" src="'+Path(r["video"]["path"]).name+'"></video>' for r in records)
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>R18任务去噪诊断</title><style>body{max-width:1300px;margin:24px auto;font:16px sans-serif}video{width:100%}</style><h1>R18：真实参考 / 冻结 / 同窗口适配 / 独有状态适配</h1><p>Tiny为Train过拟合；Main若存在也仅为历史曝光评价。AIST机制实验，不代表敦煌舞风格或Native合格。固定世界坐标，无平滑。PROGRESSIVE前75帧为共同生成历史。</p>'+cards,encoding="utf-8")
    write(out/"protocol.json",dict(code_commit=commit(),records=records,human_reviewed=False,operational_qualified=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(a.root,a.out)
