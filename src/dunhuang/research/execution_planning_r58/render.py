import subprocess
from PIL import Image,ImageDraw,ImageFont
from .common import *
def run(source,out):
 source=Path(source);out=fresh(out);d=read(source/'result.json');z=np.load(source/'arrays.npz');records={r['arm']:r for r in d['records'] if r['song']==0};arms=['B0','B1','C2'];assert all(records[a]['status']=='NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW' for a in arms);P=[z['song0_'+a+'_P'] for a in arms];N=len(P[0]);parents=z['parents'];angle=-20*np.pi/180;X=[np.cos(angle)*v[:,:,0]+np.sin(angle)*v[:,:,2] for v in P];lo=min(p[:,:,1].min() for p in P);hi=max(p[:,:,1].max() for p in P);mid=(hi+lo)/2;scale=min(440/(hi-lo),370/max(2*np.max(abs(v)) for v in X))*.92;font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);audio=Path(d['binding']['parent']['music'][0]['path']);assert sha(audio)==d['music']['0']['audio_sha256'];movie=out/'B0_B1_C2_full.mp4';log=(out/'ffmpeg.log').open('wb');cmd=['ffmpeg','-hide_banner','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x800','-r','30','-i','pipe:0','-i',str(audio),'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','20','-threads','2','-c:a','aac','-b:a','160k','-movflags','+faststart','-shortest',str(movie)];proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
 try:
  for f in range(N):
   img=Image.new('RGB',(1280,800),(244,241,232));g=ImageDraw.Draw(img);g.text((25,20),'R58 | SAME FULL MUSIC / CAMERA / SCALE | B0 fixed / B1 structure / C2 joint',font=font,fill='#173e36');g.text((25,55),f'{f/30:.3f} / {d["music"]["0"]["duration_seconds"]:.3f} sec | frame {f}/{N-1}',font=font,fill='#173e36')
   for k,arm in enumerate(arms):
    left=12+426*k;r=records[arm];region=next(s for s in r['regions'] if s['interval'][0]<=f<s['interval'][1]);event=next((e for e in r['events'] if e['target_interval'][0]<=f<e['target_interval'][1]),None);color='#1f8070' if region['kind']=='core' else '#b17b35';g.rounded_rectangle((left,100,left+402,640),radius=14,fill='white');g.text((left+10,115),arm+' | '+(event['source_uid'] if event else 'Geometry transition'),font=small,fill=color);xy=np.stack((left+201+scale*X[k][f],375-scale*(P[k][f,:,1]-mid)),axis=-1)
    for j in range(1,22):g.line([tuple(xy[int(parents[j])]),tuple(xy[j])],fill=color if j%2 else '#b17b35',width=4)
    for span in r['regions']:
     a,b=span['interval'];g.rectangle((left+4+394*a/N,660,left+4+394*b/N,673),fill='#1f8070' if span['kind']=='core' else '#d5a95d')
    g.line((left+4+394*f/(N-1),652,left+4+394*f/(N-1),680),fill='#782823',width=3)
   g.text((25,718),'Actual ChangE sources + legal time maps + frozen geometry screens. No diffusion generation.',font=small,fill='#173e36');g.text((25,750),'Body22 Root-relative; world Root/contact UNKNOWN. Native24 and human preference NOT EVALUATED.',font=small,fill='#173e36')
   if f in (0,N//2,N-1):img.save(out/f'frame_{f}.jpg',quality=92)
   proc.stdin.write(img.tobytes())
   if (f+1)%900==0:print('R58_RENDER',f+1,N,flush=True)
  proc.stdin.close();assert proc.wait()==0
 except BaseException:proc.terminate();proc.wait();raise
 finally:log.close()
 info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(movie)]));v=next(s for s in info['streams'] if s['codec_type']=='video');assert int(v['nb_frames'])==N and abs(float(info['format']['duration'])-d['music']['0']['duration_seconds'])<1/30;write(out/'result.json',dict(frames=N,fps=30,duration=float(info['format']['duration']),movie_sha256=sha(movie),source_arrays_sha256=sha(source/'arrays.npz'),audio_sha256=sha(audio),can_route=False))
