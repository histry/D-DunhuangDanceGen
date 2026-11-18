"""Full-music baseline/candidate MP4 and optional method-blind preparation media."""
import subprocess
from PIL import Image,ImageDraw,ImageFont
from .common import *

def run(source,out,blind=False):
 source=Path(source);out=fresh(out);d=read(source/'result.json');z=np.load(source/'arrays.npz');assert d['stage']=='main';records={r['arm']:r for r in d['records']};assert all(records[a]['status']=='NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW' for a in ('B0','B3'));order=['B0','B3'];rng=np.random.default_rng(CFG['seed']);order=list(rng.permutation(order)) if blind else order;P=[z['song0_'+a+'_P'] for a in order];parents=z['parents'];frames=len(P[0]);audio=Path(d['binding']['parent']['music'][0]['path']);assert sha(audio)==d['music']['0']['audio_sha256'];fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(fontpath,21);small=ImageFont.truetype(fontpath,17);title=ImageFont.truetype(fontpath,27);angle=-20*np.pi/180;x=[np.cos(angle)*p[:,:,0]+np.sin(angle)*p[:,:,2] for p in P];lo=min(p[:,:,1].min() for p in P);hi=max(p[:,:,1].max() for p in P);mid=(lo+hi)/2;scale=min(470/(hi-lo),520/max(2*np.max(abs(v)) for v in x))*.92;movie=out/('blind_pair.mp4' if blind else 'B0_vs_B3_full.mp4');log=(out/'ffmpeg.log').open('wb');cmd=['ffmpeg','-hide_banner','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x820','-r','30','-i','pipe:0','-i',str(audio),'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','20','-threads','2','-c:a','aac','-b:a','160k','-movflags','+faststart','-shortest',str(movie)];proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
 try:
  for f in range(frames):
   img=Image.new('RGB',(1280,820),(244,241,232));g=ImageDraw.Draw(img);g.text((35,22),'dunhuangwu2 | '+('A / B preference preparation' if blind else 'R56 | Frozen B0 / Structure B3'),font=title,fill=(23,62,54));g.text((35,65),f'{f/30:06.2f}s / {d["music"]["0"]["duration_seconds"]:.2f}s | frame {f}/{frames-1}',font=font,fill=(23,62,54))
   for k,arm in enumerate(order):
    left=30+640*k;r=records[arm];region=next(t for t in r['regions'] if t['interval'][0]<=f<t['interval'][1]);core=region['kind']=='core';event=next((e for e in r['events'] if e['target_interval'][0]<=f<e['target_interval'][1]),None);color=(31,128,112) if core else (177,123,53);g.rounded_rectangle((left,110,left+580,660),radius=15,fill='white');label=('A' if k==0 else 'B') if blind else arm+' | '+(event['source_uid'] if event else 'Geometry transition');g.text((left+15,125),label,font=small,fill=color);xy=np.stack((left+290+scale*x[k][f],405-scale*(P[k][f,:,1]-mid)),axis=-1)
    for j in range(1,22):g.line([tuple(xy[int(parents[j])]),tuple(xy[j])],fill=color if j%2 else (177,123,53),width=5)
    for span in r['regions']:
     a,b=span['interval'];g.rectangle((left+5+570*a/frames,680,left+5+570*b/frames,694),fill=(31,128,112) if span['kind']=='core' else (213,169,93))
    q=left+5+570*f/(frames-1);g.line((q,673,q,700),fill=(120,40,35),width=3)
   g.text((35,734),'Same full music, camera and scale. Body22 Root-relative; world Root/contact UNKNOWN.',font=small,fill=(23,62,54));g.text((35,767),'Native24 NOT EVALUABLE. No human ratings collected.' if blind else 'Geometry research; acoustic structure proposals are not annotated musical phrases.',font=small,fill=(23,62,54))
   if f in (0,frames//2,frames-1):img.save(out/f'frame_{f}.jpg',quality=92)
   proc.stdin.write(img.tobytes())
   if (f+1)%900==0:print('R56_RENDER',f+1,frames,blind,flush=True)
  proc.stdin.close();assert proc.wait()==0
 except BaseException:proc.terminate();proc.wait();raise
 finally:log.close()
 info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(movie)]));video=next(s for s in info['streams'] if s['codec_type']=='video');assert int(video['nb_frames'])==frames and abs(float(info['format']['duration'])-d['music']['0']['duration_seconds'])<1/30
 write(out/'result.json',dict(frames=frames,fps=30,source_arrays_sha256=sha(source/'arrays.npz'),movie_sha256=sha(movie),duration=float(info['format']['duration']),blind=blind,model_calls=0,optimizer_calls=0,source_audio_sha256=sha(audio)))
 if blind:
  write(out/'admin_mapping.json',dict(scope='ADMIN_ONLY_DO_NOT_SHOW_REVIEWERS',A=order[0],B=order[1]));write(out/'review_protocol.json',dict(state='PREPARATION_ONLY',human_reviewed=False,independent_records=0,media='blind_pair.mp4',media_sha256=sha(movie),questions=['naturalness','music_coordination','perceptible_difference'],choices=['A','B','SAME','UNSURE'],required_record_fields=['reviewer_id','reviewed_at','blind_item_id','media_sha256','decision','UNSURE_reason'],historically_exposed_music=True,no_ground_truth_claim=True))
 print('R56_MOVIE_COMPLETE',movie,flush=True)
