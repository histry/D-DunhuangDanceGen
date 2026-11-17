"""Fixed-metadata MP4 triptychs from actual full samples, at the original 30FPS clock."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk


def run(generation,out):
    generation=Path(generation);out=Path(out)
    if out.exists():raise ValueError('Fresh video output')
    out.mkdir(parents=True);meta=read(generation/'result.json');z=np.load(generation/'arrays.npz');J=z['skeleton_J22'];parents=z['parents'];label=meta['label'];records=[]
    uids=['female_FeiTian','male_pipa_1','male_pipa_2','male_ribbon_FenHe']
    chosen=[min((r for r in meta['records'] if r['recording_uid']==uid and r['target_music'] is None),key=lambda r:r['source_interval'][0]) for uid in uids]
    chosen += [r for r in meta['records'] if r['recording_uid']=='female_FeiTian' and r['target_music'] is not None]
    for row in chosen:
        key=row['key'];arrays=[relative_fk(z[key+'_reference_R'],J,parents),z[key+'_Frozen_22'][0],z[key+'_'+label+'RTS_22'][0]];arrays=[a-a[:,:1] for a in arrays];file=out/(key+'.mp4');cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1200x500','-r','30','-i','-']
        if row.get('audio'):cmd+=['-i',str(generation/row['audio'])]
        cmd+=['-c:v','libx264','-preset','veryfast','-crf','22','-threads','2']
        if row.get('audio'):cmd+=['-c:a','aac','-shortest']
        cmd+=[str(file)];proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);font=ImageFont.load_default()
        try:
            for t in range(256):
                image=Image.new('RGB',(1200,500),'#f4f7ef');d=ImageDraw.Draw(image);d.text((12,12),'R42 ACTUAL FULL SAMPLING / '+key,font=font,fill='#223d36');d.text((12,35),'FRAME '+str(t)+' / 255   '+('HIDDEN GAP' if 104<=t<152 else 'PROTECTED VISIBLE BODY'),font=font,fill='#a16c2b')
                for j,(title,a) in enumerate(zip(['REAL TARGET (HIDDEN NOT CONDITION)','FROZEN R39 (RAW)','FEASIBLE CONTEXT + FIXED RTS'],arrays)):
                    d.text((j*400+12,65),title,font=font,fill='#223d36');p=a[t];pts=[(round(j*400+200+(q[0]*.86-q[2]*.5)*155),round(260-q[1]*155)) for q in p]
                    for k in range(1,22):d.line([pts[int(parents[k])],pts[k]],fill='#25796e' if k%2 else '#b67c37',width=3)
                    for x,y in pts:d.ellipse((x-2,y-2,x+2,y+2),fill='#385448')
                d.text((12,470),'Same camera/scale. Root-relative common22. No fingers/contact truth. Seed '+str(row['seeds'][0]),font=font,fill='#385448');proc.stdin.write(np.asarray(image).tobytes())
        finally:proc.stdin.close()
        assert proc.wait()==0
        info=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','csv=p=0',str(file)],text=True).strip();assert info=='256',info
        records.append(dict(name=file.name,bytes=file.stat().st_size,sha256=sha(file),frames=256,role=row['role'],recording_uid=row['recording_uid'],audio=bool(row.get('audio')),seed=row['seeds'][0]));print('R42_VIDEO',key,flush=True)
    write(out/'protocol.json',dict(files=records,selection='FIXED_RECORDINGS_FIRST_INTERVAL_NOT_SCORE_SELECTED',root_relative=True,fps=30,actual_arrays_sha256=sha(generation/'arrays.npz'),whole_song_generated=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generation',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.generation,a.out)
