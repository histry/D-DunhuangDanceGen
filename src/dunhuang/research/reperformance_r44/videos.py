"""Three metadata-registered clips; all frames decoded after encoding."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from .archive import Arrays

def run(main,out):
    main=Path(main);out=Path(out)
    if out.exists():raise ValueError('Fresh videos')
    out.mkdir(parents=True);z=Arrays(main/'arrays.npz');meta=read(main/'result.json');cfg=meta['config'];records=[];J=z['skeleton_J22'];parents=z['parents'];font=ImageFont.load_default()
    for uid in cfg['registered_video_sources']:
        row=next(r for r in meta['records'] if r['source']['recording_uid']==uid);key=row['key'];tracks=[relative_fk(z[key+'_reference_R'],J,parents)]+[z[key+'_'+m+'_22'][0] for m in ('Geometry','E1W','C2','R43Core')];tracks=[p-p[:,:1] for p in tracks];file=out/(key+'.mp4');proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1800x540','-r','30','-i','-','-i',str(main/row['audio']),'-c:v','libx264','-preset','veryfast','-crf','22','-threads','2','-c:a','aac','-shortest',str(file)],stdin=subprocess.PIPE)
        try:
            for t in range(256):
                image=Image.new('RGB',(1800,540),'#f4f7ef');draw=ImageDraw.Draw(image);draw.text((12,12),'R44 / '+key+' / ACTUAL FRAME '+str(t)+' / '+('FIXED GEOMETRY BRIDGE' if 104<=t<152 else 'SOURCE CORE / PROTECTED EDGE'),font=font,fill='#223d36')
                for j,(title,points) in enumerate(zip(('ORIGINAL SOURCE / GAP EVAL ONLY','GEOMETRY','E1-W PIECEWISE CLOCK','C2 LEGAL CLOCK','SAVED R43 CORE FAILURE'),tracks)):
                    draw.text((j*360+12,55),title,font=font,fill='#223d36');pos=[(round(j*360+180+(q[0]*.86-q[2]*.5)*145),round(275-q[1]*145)) for q in points[t]]
                    for k in range(1,22):draw.line([pos[int(parents[k])],pos[k]],fill='#25796e' if k%2 else '#b67c37',width=3)
                draw.text((12,490),'Fixed camera/scale. Root-relative common22; no finger/contact/world-Root truth. R44 does not add neural motion. Geometry seed duplicates are not independent trials.',font=font,fill='#385448');proc.stdin.write(np.asarray(image).tobytes())
        finally:proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('Video encode failed')
        frames=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','csv=p=0',str(file)],text=True).strip();assert frames=='256';subprocess.run(['ffmpeg','-v','error','-i',str(file),'-f','null','-'],check=True);records.append(dict(name=file.name,bytes=file.stat().st_size,sha256=sha(file),frames=256,source_uid=uid,source_seed=row['seeds'][0],new_neural_samples=0))
    html='<!doctype html><meta charset="utf-8"><title>R44 固定来源对照视频</title><style>body{max-width:1450px;margin:30px auto;font:18px system-ui;background:#f4f7ef;color:#234}video{width:100%}</style><h1>R44：保留姿态路径，改变动作时钟</h1><p>五栏：原始来源／Geometry／E1-W／C²／保存的R43失败样本。每段256帧，同音乐、同相机、同尺度。R44为几何重定时，没有新增扩散采样；原始来源缺口只作评价。</p><p><a href="../viewer/index.html">全部22案例：共同音乐时钟、逐帧、诊断与时间图</a></p>'+''.join('<h2>'+r['source_uid']+'</h2><video controls preload="metadata" src="'+r['name']+'"></video>' for r in records)
    (out/'index.html').write_text(html,encoding='utf8');write(out/'protocol.json',dict(selection='THREE_REGISTERED_SOURCE_UIDS_NOT_SCORE_SELECTED',files=records+[dict(name='index.html',bytes=(out/'index.html').stat().st_size,sha256=sha(out/'index.html'))]));z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.main,a.out)
