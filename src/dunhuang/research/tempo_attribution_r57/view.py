import base64,io
import soundfile as sf
from .common import *
from dunhuang.research.strong_baseline_r52.view import finish,encode
NAMES=dict(Native='来源原速 · 无配对音乐真值',B0='固定槽位 · 原 E1-W',LinearB1='结构槽位 · 匀速重定时',B1='结构槽位 · 原 E1-W')
def viewer(out,result,arrays,pop):
 folder=fresh(out/'viewer');cases=[]
 for r in result['records']:
  frames=max(m['frames'] for m in r['methods'].values());wave,sr=sf.read(pop.tracks[r['song']]['path'],frames=round(frames/30*sf.info(pop.tracks[r['song']]['path']).samplerate));buffer=io.BytesIO();sf.write(buffer,wave,sr,format='WAV');methods=[]
  for name,info in r['methods'].items():
   p=arrays[r['key']+'_'+name+'_P'];padded=np.zeros((frames,22,3));padded[:len(p)]=p;jerk=np.r_[np.zeros(3),np.sqrt(np.mean((np.diff(p,n=3,axis=0)*27000)**2,axis=(1,2)))];methods.append(dict(name=NAMES[name],data=encode(padded),jerk=jerk.tolist(),observed_frames=len(p)))
  cases.append(dict(label=r['source_uid']+' · '+r['event_uid'],frames=frames,origin=0,audio='data:audio/wav;base64,'+base64.b64encode(buffer.getvalue()).decode(),methods=methods,meta=dict(regions=[],scope='同一完整来源、骨架、30FPS和首槽音乐时钟。各栏保持实际长度，结束后隐藏，不循环或冻结姿态。原速来源无音乐配对真值。D3不是自然性结论；Root/接触未知。',**r)))
 finish(folder,cases,pop.net.parents);p=folder/'index.html';text=p.read_text(encoding='utf8').replace('R52','R57').replace('保留 Heuristic 强基线；生成比较同一完整来源与几何基线，修复评价时间坐标。来源阶段为观测运动学代理，世界 Root／接触及 Native24 尚未取得完整资格。','先观察同一事件的原速、固定槽位、结构槽位匀速和E1-W。这里是受控重定时轨迹，没有新神经生成或质量升级结论。').replace("g.clearRect(0,0,320,360);const xy=", "g.clearRect(0,0,320,360);if(frame>=row.methods[i].observed_frames){g.fillStyle='#777';g.fillText('实际支持已结束 / END OF SUPPORT',15,170);return}const xy=")
 p.write_bytes(text.encode());write(folder/'protocol.json',dict(cases=len(cases),fps=30,no_loop_or_pose_hold_padding=True,files=[dict(name=q.name,sha256=sha(q),bytes=q.stat().st_size) for q in sorted(folder.iterdir()) if q.name!='protocol.json']))

def movie(source,out):
 import subprocess
 from PIL import Image,ImageDraw,ImageFont
 source=Path(source);out=fresh(out);data=read(source/'result.json');z=np.load(source/'arrays.npz');parents=z['parents'];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);rows=[]
 labels=['Native original tempo','B0 fixed / E1-W','B1 length / uniform','B1 length / E1-W']
 for r in data['records']:
  P=[z[r['key']+'_'+name+'_P'] for name in r['methods']];frames=max(map(len,P));angle=-20*np.pi/180;X=[np.cos(angle)*p[:,:,0]+np.sin(angle)*p[:,:,2] for p in P];lo=min(p[:,:,1].min() for p in P);hi=max(p[:,:,1].max() for p in P);mid=(lo+hi)/2;scale=min(360/(hi-lo),270/max(2*np.max(abs(x)) for x in X))*.9;audio=Path(data['parent']['music'][r['song']]['path']);dest=out/(r['key']+'.mp4');log=(out/(r['key']+'.log')).open('wb');cmd=['ffmpeg','-hide_banner','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x680','-r','30','-i','pipe:0','-i',str(audio),'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','ultrafast','-crf','23','-threads','2','-c:a','aac','-b:a','128k','-movflags','+faststart','-t',str(frames/30),str(dest)];proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log,stdout=subprocess.DEVNULL)
  try:
   for f in range(frames):
    image=Image.new('RGB',(1280,680),(244,241,232));g=ImageDraw.Draw(image);g.text((25,20),'R57 | SAME SOURCE / DIFFERENT LEGAL TIME MAPS | '+r['source_uid'],font=font,fill='#173e36');g.text((25,55),f'{f/30:.3f} sec | actual frame {f} | native source is NOT paired to this music',font=small,fill='#173e36')
    for k,p in enumerate(P):
     left=k*320+8;g.rounded_rectangle((left,100,left+304,540),radius=12,fill='white');g.text((left+7,115),labels[k],font=small,fill='#173e36')
     if f>=len(p):g.text((left+20,300),'END OF ACTUAL SUPPORT',font=small,fill='#777');continue
     xy=np.stack((left+152+scale*X[k][f],325-scale*(p[f,:,1]-mid)),axis=-1)
     for j in range(1,22):g.line([tuple(xy[int(parents[j])]),tuple(xy[j])],fill='#1f8070' if j%2 else '#b17b35',width=4)
     name=list(r['methods'])[k];m=r['methods'][name]['metrics'];ratio=r['methods'][name]['amplification']['3']['ratio'];g.text((left+8,555),f'N={len(p)} D3q95={m["D3_Q95"]:.2f}',font=small,fill='#173e36');g.text((left+8,581),'A3='+('UNDEFINED' if ratio is None else f'{ratio:.3f}'),font=small,fill='#173e36')
    g.text((25,627),'Actual source retention, not diffusion. Body22 Root-relative; Native24 NOT EVALUABLE.',font=small,fill='#173e36');proc.stdin.write(image.tobytes())
   proc.stdin.close();assert proc.wait()==0
  except BaseException:proc.terminate();proc.wait();raise
  finally:log.close()
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(dest)]));v=next(x for x in info['streams'] if x['codec_type']=='video');assert int(v['nb_frames'])==frames;rows.append(dict(path=dest.name,frames=frames,sha256=sha(dest),duration=float(info['format']['duration']),source_key=r['key']));print('R57_VIDEO',dest.name,frames,flush=True)
 write(out/'result.json',dict(files=rows,source_result_sha256=sha(source/'result.json'),source_arrays_sha256=sha(source/'arrays.npz'),native_not_music_paired=True,can_route=False));(out/'index.html').write_bytes(('<!doctype html><meta charset="utf-8"><h1>R57 同源时间映射实算</h1><p>原速来源没有真实音乐配对；实际支持结束后隐藏。这里不是新模型生成。</p>'+''.join('<p>'+r['source_uid']+'</p><video controls width="100%" src="'+row['path']+'"></video>' for r,row in zip(data['records'],rows))).encode())
