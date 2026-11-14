"""Fixed-world Source/D0/N0/N1/N2[/U], Train-only, no smoothing."""
import argparse,math,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.objective_r19.common import population

def run(root,out):
    root=Path(root);out=fresh(out);cfg=read(root/"calibration/protocol.json")["config"]
    pre,rows=population(cfg);native_environment(pre["assets"])
    from motion.geometry import fk_24_np
    from motion_geometry.smpl24 import PARENTS
    mapping={(b["case"],b["arm"]):root/"native"/folder for folder,b in read(root/"native/protocol.json")["saved"].items() if b["control"]=="correct"}
    arms=["D0"]+cfg["arms"]+(["U"] if read(root/"gate.json")["unique_admitted"] else [])
    width=(len(arms)+1)*300;records=[];font=ImageFont.load_default();c,s=math.cos(-.45),math.sin(-.45)
    def project(x):return np.stack((c*x[...,0]+s*x[...,2],x[...,1]-.18*(-s*x[...,0]+c*x[...,2])),-1)
    for row in rows:
        case=row["case"];z=np.load(checked(row["arrays"]));tracks=[fk_24_np(z["native"])]+[np.load(mapping[case,arm]/"fk.npy") for arm in arms]
        xy=np.stack([project(x) for x in tracks]);lo=xy.min((0,1,2));hi=xy.max((0,1,2));center=(lo+hi)/2
        scale=min(260/max(float(hi[0]-lo[0]),1e-6),450/max(float(hi[1]-lo[1]),1e-6))
        path=out/f"case{case}.mp4";audio=checked(row["audio225"])
        proc=subprocess.Popen(["ffmpeg","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x600","-r","30","-i","-",
            "-i",str(audio),"-map","0:v:0","-map","1:a:0","-c:v","libx264","-pix_fmt","yuv420p","-preset","veryfast",
            "-crf","20","-threads","2","-c:a","aac","-shortest",str(path)],stdin=subprocess.PIPE)
        try:
            for t in range(225):
                im=Image.new("RGB",(width,600),"#f2f4f7");draw=ImageDraw.Draw(im)
                for k,label in enumerate(["Paired AIST reference"]+arms):
                    off=k*300;draw.text((off+10,12),label,fill="#173249",font=font)
                    draw.text((off+10,32),f"Train{case} inbetween | {t+1}/225",fill="#40556c",font=font)
                    pts=[(round(off+150+(v[0]-center[0])*scale),round(315-(v[1]-center[1])*scale)) for v in xy[k,t]]
                    for j in range(1,24):draw.line([pts[int(PARENTS[j])],pts[j]],fill="#285e99" if j%2 else "#9c5133",width=3)
                    for a,b in pts:draw.ellipse((a-2,b-2,a+2,b+2),fill="#22394c")
                    draw.text((off+10,553),"TRAIN MECHANISM / UNQUALIFIED",fill="#415b6b",font=font)
                    draw.text((off+10,575),"Fixed world / no smoothing",fill="#415b6b",font=font)
                proc.stdin.write(np.asarray(im).tobytes())
        finally:proc.stdin.close()
        assert proc.wait()==0
        frames=int(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","csv=p=0",str(path)],text=True).strip())
        assert frames==225;subprocess.run(["ffmpeg","-v","error","-i",str(path),"-f","null","-"],check=True)
        records.append(dict(case=case,video=ref(path),frames=frames,full_decode_passed=True,audio=row["audio225"]))
    cards="".join('<h2>Train '+str(r["case"])+'</h2><video controls preload="metadata" src="'+Path(r["video"]["path"]).name+'"></video>' for r in records)
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>R19目标消融</title><style>body{max-width:1800px;margin:24px auto;font:16px sans-serif}video{width:100%}</style><h1>R19：首尾补全目标消融</h1><p>Source / D0冻结 / N0原目标 / N1基础重建 / N2静态校准 / U若准入。仅两段AIST Train机制，225帧，真实边界；不是敦煌风格成果或物理合格声明。保留世界Root，无平滑。</p>'+cards,encoding="utf-8")
    write(out/"protocol.json",dict(code_commit=commit(),records=records,human_reviewed=False,operational_qualified=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(a.root,a.out)
