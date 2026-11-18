"""Fixed primary bridge clips; actual arrays and explicit rejection, no model sampling."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.reperformance_r44.archive import Arrays

def run(main,out):
    main=Path(main);out=Path(out)
    if out.exists():raise ValueError('Fresh primary videos')
    out.mkdir(parents=True);meta=read(main/'result.json');z=Arrays(main/'arrays.npz');J=z['skeleton_J22'];parents=z['parents'];files=[];font=ImageFont.load_default()
    for key in meta['config']['registered_video_cases']:
        r=next(r for r in meta['records'] if r['key']==key);tracks=[relative_fk(z[key+'_reference_R'],J,parents),z[key+'_G0_22'][0],z[key+'_G1_22'][0] if 'G1' in r['methods'] else None];tracks=[None if p is None else p-p[:,:1] for p in tracks];file=out/(key+'.mp4');proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1200x540','-r','30','-i','-','-i',str(main/r['audio']),'-c:v','libx264','-preset','veryfast','-crf','22','-threads','2','-c:a','aac','-shortest',str(file)],stdin=subprocess.PIPE)
        try:
            for t in range(256):
                image=Image.new('RGB',(1200,540),'#f4f7ef');draw=ImageDraw.Draw(image);draw.text((12,12),'R47 / '+key+' / FRAME '+str(t),font=font,fill='#234')
                for i,(title,P) in enumerate(zip(('SOURCE / GAP EVALUATION ONLY','G0 FROZEN QUINTIC','G1 REANCHORED SO3 / NO TRAINING'),tracks)):
                    draw.text((i*400+12,50),title,font=font,fill='#234')
                    if P is None:draw.text((i*400+50,200),'REJECTED / NO ELIGIBLE BRIDGE',font=font,fill='#a33');continue
                    pos=[(round(i*400+200+(v[0]*.86-v[2]*.5)*150),round(275-v[1]*150)) for v in P[t]]
                    for j in range(1,22):draw.line([pos[int(parents[j])],pos[j]],fill='#25796e' if j%2 else '#b67c37',width=3)
                draw.text((12,490),'Same scale/camera/time. Common22 Root-relative. Geometry admission is not Native or artistic quality.',font=font,fill='#345');proc.stdin.write(np.asarray(image).tobytes())
        finally:proc.stdin.close()
        assert proc.wait()==0;assert subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','csv=p=0',str(file)],text=True).strip()=='256';subprocess.run(['ffmpeg','-v','error','-i',str(file),'-f','null','-'],check=True);files.append(dict(name=file.name,bytes=file.stat().st_size,sha256=sha(file),frames=256))
    html='<!doctype html><meta charset="utf-8"><title>R47 实际桥接对照</title><style>body{max-width:1300px;margin:30px auto;background:#f4f7ef;color:#234;font:18px system-ui}video{width:100%}</style><h1>冻结五次桥与重新锚定 SO(3) 桥</h1><p>三栏：来源／G0／G1。256帧、同音乐与相机。两侧观察相同，保护区和Root/contact保留。没有新神经采样；图域合法不等于自然或物理合格，异源连接没有唯一隐藏真值。</p>'+''.join('<h2>'+f['name']+'</h2><video controls preload="metadata" src="'+f['name']+'"></video>' for f in files);(out/'index.html').write_text(html,encoding='utf8');files.append(dict(name='index.html',bytes=(out/'index.html').stat().st_size,sha256=sha(out/'index.html')));write(out/'protocol.json',dict(files=files,new_neural_samples=0,selection='FOUR_REGISTERED_CASE_IDENTITIES_NOT_R47_SCORE_SELECTION'));z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.main,a.out)
