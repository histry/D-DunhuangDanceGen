"""Four identity-registered source/G0/G3 clips, actual arrays and no invented audio."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from dunhuang.research.complete_r36.core import read,write,sha

def run(main,out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh videos')
    out.mkdir(parents=True);main=Path(main);z=np.load(main/'arrays.npz');meta=read(main/'result.json');files=[]
    for key in ('female_FeiTian_0','female_meditation_0','male_pipa_2_0','cross_00'):
        r=next(r for r in meta['records'] if r['key']==key);file=out/(key+'.mp4');proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1200x540','-r','30','-i','-','-c:v','libx264','-preset','veryfast','-crf','22','-threads','2',str(file)],stdin=subprocess.PIPE)
        for t in range(256):
            im=Image.new('RGB',(1200,540),'#f5f3ec');d=ImageDraw.Draw(im);d.text((15,15),'R48 / '+key+' / frame '+str(t),fill='#234')
            for k,name in enumerate(('source','G0','G3')):
                d.text((k*400+15,50),name+' / '+('gap' if 104<=t<152 else 'protected'),fill='#234');p=z[key+'_source'] if name=='source' else z[key+'_'+name+'_P'] if name in r['methods'] else None
                if p is None or (name=='source' and not z[key+'_source_valid'][t]):d.text((k*400+35,220),'NO TRUTH / REJECTED',fill='#a43');continue
                points=[(int(k*400+200+(v[0]*.86-v[2]*.5)*150),int(275-v[1]*150)) for v in p[t]]
                for j in range(1,22):d.line([points[int(z['parents'][j])],points[j]],fill='#24796f' if j%2 else '#b77735',width=3)
            d.text((15,495),'Actual saved geometry. Root-relative common22. No paired music, contact truth, or Native qualification.',fill='#234');proc.stdin.write(np.asarray(im).tobytes())
        proc.stdin.close();assert proc.wait()==0;assert subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','csv=p=0',str(file)],text=True).strip()=='256';subprocess.run(['ffmpeg','-v','error','-i',str(file),'-f','null','-'],check=True);files.append(dict(name=file.name,bytes=file.stat().st_size,sha256=sha(file),frames=256))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>R48 实际桥接视频</title><style>body{max-width:1200px;margin:30px auto;font:18px system-ui;background:#f5f3ec}video{width:100%}</style><h1>来源／G0／G3：新身份实际轨迹</h1><p>不新增神经动作。异源来源缺口留空，没有伪造配乐或自然性结论。每条256帧，同一尺度与相机。</p>'+''.join('<h2>'+f['name']+'</h2><video controls preload="metadata" src="'+f['name']+'"></video>' for f in files),encoding='utf8');p=out/'index.html';files.append(dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)));write(out/'protocol.json',dict(files=files,neural_samples=0,paired_music=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.main,a.out)
