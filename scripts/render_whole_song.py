"""Render saved full-song body22 trajectories without invoking a motion model."""
import argparse,json,hashlib,subprocess,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('outputs/whole_song_geometry_02/main'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);meta=read(a.source/'result.json');assert meta['status']=='GEOMETRY_WHOLE_SONG_RESEARCH_PREVIEW';arrays_sha=sha(a.source/'arrays.npz');z=np.load(a.source/'arrays.npz');P=z['positions_root_relative'].astype(float);parents=z['parents'];assert P.shape==(meta['frames'],22,3) and np.isfinite(P).all();audio=Path(meta['audio']);assert sha(audio)==meta['audio_sha256'];width,height=1280,800
 font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf');font=ImageFont.truetype(str(font_path),21);small=ImageFont.truetype(str(font_path),17);title=ImageFont.truetype(str(font_path),29);angles=[math.radians(-20),math.radians(60)];x=[np.cos(t)*P[:,:,0]+np.sin(t)*P[:,:,2] for t in angles];y=P[:,:,1];mid=(y.max()+y.min())/2;scale=min(470/max(np.ptp(y),1e-8),520/max(2*max(np.max(abs(v)) for v in x),1e-8))*.92
 movie=a.out/'dunhuangwu2_geometry_full.mp4';log=(a.out/'ffmpeg.log').open('wb');cmd=['ffmpeg','-hide_banner','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','30','-i','pipe:0','-i',str(audio),'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','20','-threads','2','-c:a','aac','-b:a','160k','-movflags','+faststart','-shortest',str(movie)];proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log);spans=meta['regions'];slot=0
 try:
  for f in range(len(P)):
   while slot<len(spans)-1 and f>=spans[slot]['interval'][1]:slot+=1
   region=spans[slot];core=region['kind']=='core';event_index=slot//2;label=meta['selected'][event_index]['source_uid'] if core else 'Geometry transition '+str(event_index+1)+' -> '+str(event_index+2);color=(31,128,112) if core else (183,123,49)
   img=Image.new('RGB',(width,height),(244,241,232));d=ImageDraw.Draw(img);d.text((40,25),'dunhuangwu2 | ChangE + E1-W + G0/G1',font=title,fill=(23,62,54));d.text((40,74),f'{f/30:06.2f}s / {meta["duration_seconds"]:.2f}s | frame {f}/{len(P)-1} | '+label,font=font,fill=color)
   for view in range(2):
    left=30+640*view;d.rounded_rectangle((left,125,left+580,655),radius=15,fill='white');d.text((left+18,140),'Root-relative view '+str(view+1),font=small,fill=(23,62,54));xy=np.stack((left+290+scale*x[view][f],405-scale*(P[f,:,1]-mid)),axis=-1)
    for j in range(1,22):
     d.line([tuple(xy[int(parents[j])]),tuple(xy[j])],fill=color if j%2 else (177,123,53),width=5);cx,cy=xy[j];d.ellipse((cx-3,cy-3,cx+3,cy+3),fill=(23,62,54))
   d.text((40,674),'Geometry research preview; no Diffusion/SDMR. World Root/contact UNKNOWN.',font=small,fill=(23,62,54));d.text((40,701),'Body22 local pose only; Native24 NOT EVALUABLE. All 120 seconds, 30 FPS.',font=small,fill=(23,62,54))
   for r in spans:
    aa,bb=r['interval'];d.rectangle((40+1200*aa/len(P),740,40+1200*bb/len(P),753),fill=(31,128,112) if r['kind']=='core' else (213,169,93))
   pos=40+1200*f/(len(P)-1);d.line((pos,732,pos,765),fill=(110,35,35),width=3)
   if f in (0,2360,len(P)-1):img.save(a.out/f'frame_{f}.jpg',quality=92)
   proc.stdin.write(img.tobytes())
   if (f+1)%600==0:print('RENDER',f+1,len(P),flush=True)
  proc.stdin.close();assert proc.wait()==0
 except BaseException:
  proc.terminate();proc.wait();raise
 finally:log.close()
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(movie)]));v=next(s for s in probe['streams'] if s['codec_type']=='video');sound=next(s for s in probe['streams'] if s['codec_type']=='audio');assert int(v['nb_frames'])==len(P) and v['r_frame_rate']=='30/1' and abs(float(probe['format']['duration'])-meta['duration_seconds'])<1/30;assert sha(a.source/'arrays.npz')==arrays_sha
 result=dict(frames=len(P),fps=30,video_duration=float(probe['format']['duration']),audio_duration=float(sound['duration']),arrays_sha256=arrays_sha,audio_source_sha256=sha(audio),video_sha256=sha(movie),renderer_sha256=sha(__file__),command=cmd,model_calls=0,optimizer_calls=0,root_relative=True,physical_qualified=False,ffprobe=probe);(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print('RENDER_COMPLETE',movie,movie.stat().st_size,flush=True)
if __name__=='__main__':main()
