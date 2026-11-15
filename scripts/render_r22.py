"""Reuse fixed-world R21 renderer with R22 labels and its explicit arm list."""
import argparse
from pathlib import Path
import json,math,subprocess
import numpy as np
from PIL import Image,ImageDraw
from render_r21 import run as render
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read

def run(root,out):
    render(root,out)
    page=Path(out)/"index.html"
    text=page.read_text(encoding="utf8").replace("R21","R22").replace("边界、内部时序与支撑机制","分支监督与真实采样").replace("rs.value='B3'","rs.value='G2'")
    page.write_text(text,encoding="utf8")
    render_lodge(Path(root),Path(out))
def render_lodge(root,out):
    result=read(root/"lodge/result.json");z=np.load(checked(result["arrays"]));parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
    c,t=math.cos(-.45),math.sin(-.45)
    def project(x):return np.stack((c*x[...,0]+t*x[...,2],x[...,1]-.18*(-t*x[...,0]+c*x[...,2])),-1)
    media=[];panels=[]
    for i,binding in enumerate(result["bindings"]):
        case=binding["case"];tracks={"Source22":z["source22"][i]}
        for mode in ("native_unconditioned","native_primitives","mismatched_primitives"):tracks[mode]=z[mode+"_22"][i]
        allxy=np.stack([project(v) for v in tracks.values()]);lo=allxy.min((0,1,2));hi=allxy.max((0,1,2));center=(lo+hi)/2
        scale=min(540/max(float(hi[0]-lo[0]),1e-6),440/max(float(hi[1]-lo[1]),1e-6));items=[]
        for mode,points in tracks.items():
            path=out/f"lodge_case{case}_{mode}.mp4";points=project(points)
            proc=subprocess.Popen(["ffmpeg","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","600x600","-r","30","-i","-","-i",str(checked(binding["audio"])),"-map","0:v:0","-map","1:a:0","-c:v","libx264","-pix_fmt","yuv420p","-preset","veryfast","-crf","20","-threads","2","-c:a","aac","-shortest",str(path)],stdin=subprocess.PIPE)
            try:
                for frame in range(256):
                    image=Image.new("RGB",(600,600),"#f2f4f7");draw=ImageDraw.Draw(image)
                    draw.text((12,12),f"Lodge official | {binding['role']} case{case} | {mode}",fill="#173249")
                    draw.text((12,34),"Native24 NOT EVALUABLE | common22 | AIST transfer mechanism",fill="#70402e")
                    pts=[(round(300+(v[0]-center[0])*scale),round(310-(v[1]-center[1])*scale)) for v in points[frame]]
                    for j in range(1,22):draw.line([pts[parents[j]],pts[j]],fill="#285e99" if j%2 else "#9c5133",width=4)
                    for a,b in pts:draw.ellipse((a-3,b-3,a+3,b+3),fill="#22394c")
                    draw.text((12,553),f"{frame+1}/256 | fixed world camera / common scale / no smoothing",fill="#415b6b")
                    draw.text((12,576),"Soft primitive cues; no known future Root / no full qualification",fill="#415b6b")
                    proc.stdin.write(np.asarray(image).tobytes())
            finally:proc.stdin.close()
            assert proc.wait()==0
            frames=int(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","csv=p=0",str(path)],text=True).strip());assert frames==256
            subprocess.run(["ffmpeg","-v","error","-i",str(path),"-f","null","-"],check=True)
            media.append(dict(case=case,mode=mode,video=ref(path),frames=256,full_decode_passed=True,full_native_status="NOT_EVALUABLE"))
            items.append(f'<div><h3>{mode}</h3><video class="case{case}" controls preload="metadata" '+('' if mode=="Source22" else 'muted ')+f'src="{path.name}"></video></div>')
        panels.append(f'<section><h2>case {case} / {binding["role"]}</h2><button onclick="toggle(\'case{case}\')">本来源同步播放／暂停</button><div class="grid">'+''.join(items)+'</div></section>')
    page="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>R22 Lodge共同骨架对比</title>
<style>body{font:16px system-ui;margin:24px;background:#f2f4f7;color:#173249}.grid{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:18px}video{width:100%}button{padding:10px}section{margin:30px 0}</style>
<h1>R22：Lodge Local 原生短段基线</h1><p>固定官方模型，139D／22关节／256帧／35D音乐。此处仅为共同骨架诊断，缺少两个Native关节，不可评价完整Native，不是敦煌舞成品。源姿态使用官方骨架与记录的坐标转换；没有平滑或逐帧居中。每个来源独立同步播放，音频来自对应真实时钟片段。</p>CONTENT
<script>function toggle(name){const vs=[...document.querySelectorAll('video.'+name)],a=vs[0];if(a.paused){for(const v of vs){v.currentTime=a.currentTime;v.play().catch(()=>{});}}else{for(const v of vs)v.pause();}}</script></html>"""
    (out/"lodge_index.html").write_text(page.replace("CONTENT",''.join(panels)),encoding="utf8")
    write(out/"lodge_protocol.json",dict(records=media,full_native_status="NOT_EVALUABLE",live_controls_verified=False))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(a.root,a.out)
