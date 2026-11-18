"""Fixed three-source actual-motion preview before factor controls."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.reperformance_r44.archive import Arrays

def run(main,out):
    main=Path(main);out=Path(out)
    if out.exists():raise ValueError('Fresh videos')
    out.mkdir(parents=True);z=Arrays(main/'arrays.npz');meta=read(main/'result.json');files=[];J=z['skeleton_J22'];parents=z['parents'];font=ImageFont.load_default()
    for uid in meta['config']['registered_video_sources']:
        row=next(r for r in meta['records'] if r['source']['recording_uid']==uid);key=row['key'];tracks=[relative_fk(z[key+'_reference_R'],J,parents)]+[z[key+'_'+m+'_22'][0] for m in ('Geometry','ObservedRebuild','Phase')];tracks=[p-p[:,:1] for p in tracks];file=out/(key+'.mp4');proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1440x540','-r','30','-i','-','-i',str(main/row['audio']),'-c:v','libx264','-preset','veryfast','-crf','22','-threads','2','-c:a','aac','-shortest',str(file)],stdin=subprocess.PIPE)
        try:
            for t in range(256):
                image=Image.new('RGB',(1440,540),'#f4f7ef');draw=ImageDraw.Draw(image);draw.text((12,12),'R46 / '+key+' / FRAME '+str(t)+' / '+('FIXED GEOMETRY BRIDGE' if 104<=t<152 else 'CORE / PROTECTED EDGE'),font=font,fill='#234')
                for j,(title,points) in enumerate(zip(('ORIGINAL SOURCE / GAP EVAL ONLY','FROZEN GEOMETRY','OBSERVED-ONLY REBUILD','FROZEN R45 PHASE'),tracks)):
                    draw.text((j*360+12,55),title,font=font,fill='#234');pos=[(round(j*360+180+(v[0]*.86-v[2]*.5)*145),round(275-v[1]*145)) for v in points[t]]
                    for k in range(1,22):draw.line([pos[int(parents[k])],pos[k]],fill='#25796e' if k%2 else '#b67c37',width=3)
                draw.text((12,490),'Same clock, camera, scale. Common22 Root-relative. Geometry clock selection, not new neural sampling. Source-feasible filter is not Native qualification.',font=font,fill='#345');proc.stdin.write(np.asarray(image).tobytes())
        finally:proc.stdin.close()
        assert proc.wait()==0;frames=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','csv=p=0',str(file)],text=True).strip();assert frames=='256';subprocess.run(['ffmpeg','-v','error','-i',str(file),'-f','null','-'],check=True);files.append(dict(name=file.name,bytes=file.stat().st_size,sha256=sha(file),frames=256,source_uid=uid))
    html='<!doctype html><meta charset="utf-8"><title>R46 主候选视频</title><style>body{max-width:1500px;margin:30px auto;font:18px system-ui;background:#f4f7ef;color:#234}video{width:100%}</style><h1>R46 冻结几何桥的身体构形责任审计</h1><p>四栏：原始来源／冻结Geometry／仅观察端点重建／冻结R45 Phase。256帧，同音乐、视角与尺度；无新增神经采样。</p><p><a href="../viewer/index.html">全部22案例逐帧与时间图</a></p>'+''.join('<h2>'+f['source_uid']+'</h2><video controls preload="metadata" src="'+f['name']+'"></video>' for f in files);(out/'index.html').write_text(html,encoding='utf8');files.append(dict(name='index.html',bytes=(out/'index.html').stat().st_size,sha256=sha(out/'index.html')));write(out/'protocol.json',dict(files=files,selection='THREE_REGISTERED_UIDS_NOT_SCORE_SELECTED',new_neural_samples=0));z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.main,a.out)
