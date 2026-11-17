"""Metadata-first actual four-way videos; no model or score-based selection."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk


def run(main,out):
    main=Path(main);out=Path(out)
    if out.exists():raise ValueError('Fresh videos')
    out.mkdir(parents=True);z=np.load(main/'arrays.npz');meta=read(main/'result.json');records=[];J=z['skeleton_J22'];parents=z['parents'];font=ImageFont.load_default()
    for uid in ('female_FeiTian','male_pipa_2'):
        row=next(r for r in meta['records'] if r['recording_uid']==uid);key=row['key'];arrays=[relative_fk(z[key+'_reference_R'],J,parents)]+[z[key+'_'+name+'_22'][0] for name in ('Geometry','Frozen','Core')];arrays=[p-p[:,:1] for p in arrays];file=out/(key+'.mp4')
        cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1440x520','-r','30','-i','-']
        if row['audio']:cmd+=['-i',str(main/row['audio'])]
        cmd+=['-c:v','libx264','-preset','veryfast','-crf','22','-threads','2']
        if row['audio']:cmd+=['-c:a','aac','-shortest']
        cmd+=[str(file)];proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for t in range(256):
                image=Image.new('RGB',(1440,520),'#f4f7ef');draw=ImageDraw.Draw(image);draw.text((12,12),'R43 ACTUAL FROZEN LODGE / '+key+' / FRAME '+str(t),font=font,fill='#223d36')
                for j,(title,points) in enumerate(zip(('REAL SOURCE / HIDDEN GAP NOT INPUT','GEOMETRY','FROZEN BRIDGE','REGIONAL CORE'),arrays)):
                    draw.text((j*360+12,55),title,font=font,fill='#223d36');pos=[(round(j*360+180+(q[0]*.86-q[2]*.5)*145),round(275-q[1]*145)) for q in points[t]]
                    for k in range(1,22):draw.line([pos[int(parents[k])],pos[k]],fill='#25796e' if k%2 else '#b67c37',width=3)
                draw.text((12,480),'Same camera, scale, seed. Root-relative common22. Core upper <=0.1rad / bridge <=0.06rad / Root <=0.03m. Not paired ChangE music.',font=font,fill='#385448');proc.stdin.write(np.asarray(image).tobytes())
        finally:proc.stdin.close()
        assert proc.wait()==0
        frames=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','csv=p=0',str(file)],text=True).strip();assert frames=='256'
        records.append(dict(name=file.name,bytes=file.stat().st_size,sha256=sha(file),frames=256,source_uid=uid,seed=row['seeds'][0],paired_music=False))
    write(out/'protocol.json',dict(files=records,selection='TWO_REGISTERED_SOURCE_UIDS_FIRST_INTERVAL_NOT_SCORE_SELECTED'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.main,a.out)
