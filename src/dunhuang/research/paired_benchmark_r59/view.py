"""True raw-grid recorded actions and frozen retrieval decisions; never called neural generation."""
import base64,subprocess,json
import soundfile as sf
from .common import *
from dunhuang.research.strong_baseline_r52.view import finish,encode

def viewer(out,data,result):
 folder=fresh(Path(out)/'viewer');cases=[];parents=np.array(data.sk['parents']);J=np.array(data.sk['J']);arrays={};details=[]
 for k,row in enumerate(result['viewer']):
  q=row['query'];qr=data.rows[q];source=data.sources[qr['source_uid']];a,b=qr['audio_interval'];audio='case'+str(k)+'.wav'
  with sf.SoundFile(checked(source['audio'])) as f:
   sr=f.samplerate;f.seek(round(a*sr/30));wave=f.read(round(b*sr/30)-round(a*sr/30));assert len(wave)==round(b*sr/30)-round(a*sr/30)
  sf.write(folder/audio,wave,sr);panels=[];identities=[]
  for name,i in [('实际配对录制参考',q),('Heuristic 检索第一',row['top']['Heuristic']),('Frozen 检索第一',row['top']['Frozen'])]:
   r=data.rows[i];z=data.source(r['source_uid']);lo,hi=r['source_interval'];P=relative_fk(z['R'][lo:hi],J,parents);arrays[f'case{k}_'+str(len(panels))]=P.astype('<f4');panels.append(dict(name=name,data=encode(P),jerk=[]));identities.append(dict(label=name,row=i,source_uid=r['source_uid'],source_interval=[lo,hi],audio_interval=r['audio_interval'],raw_motion_sha256=r['raw_motion_sha256']))
  info=dict(dataset=qr['dataset'],role=qr['role'],query=q,query_uid=qr['uid'],pool_size=len(row['pool']),pool_sha256=identity([data.rows[i]['uid'] for i in row['pool']]),sources=identities,source_clock_scope=source['clock_scope'],generated=False,metadata_selected=True,not_best_case_selection=True,root_view='ROOT_RELATIVE_DISPLAY_ONLY_RAW_TRANSLATION_PRESERVED_IN_SOURCE_ARRAYS')
  cases.append(dict(label=qr['dataset']+' · '+qr['group']+' · '+qr['role'],frames=CFG['frames'],origin=a,audio=audio,methods=panels,meta=dict(scope='R59真实配对来源回放＋完整固定网格检索；不是新舞蹈生成。AIST跨录制候选，FineDance同录制时段目标；历史曝光／官方Train诊断，不是新盲测。',regions=[],details=info)));details.append(info)
 finish(folder,cases,parents);page=(folder/'index.html').read_text(encoding='utf8').replace('R52','R59').replace('保留 Heuristic 强基线；生成比较同一完整来源与几何基线，修复评价时间坐标。来源阶段为观测运动学代理，世界 Root／接触及 Native24 尚未取得完整资格。','实际配对录制／启发式第一／冻结模型第一，共同音乐时钟与相同身体尺度。候选来自完整录制固定滑窗，不按正例插入；不是扩散生成或艺术正确率。').replace('相对身体 D3 诊断曲线（不是 Native Jerk；点击跳转）。高值须结合来源活动与动作阶段判断。','本轮只检查配对来源、候选池和时钟；空诊断曲线不代表零抖动或物理合格。')
 (folder/'index.html').write_bytes(page.encode());save_arrays(out/'viewer_arrays.npz',dict(parents=parents,**arrays));write(out/'viewer_cases.json',details);write(folder/'protocol.json',dict(cases=len(cases),generated=False,files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.iterdir()) if p.name!='protocol.json']))

def render(source,out):
 from PIL import Image,ImageDraw,ImageFont
 source=Path(source);out=fresh(out);z=np.load(source/'viewer_arrays.npz');details=read(source/'viewer_cases.json');parents=z['parents'];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);movies=[]
 for k,r in enumerate(details):
  P=[z[f'case{k}_{i}'] for i in range(3)];angle=-20*np.pi/180;X=[np.cos(angle)*v[:,:,0]+np.sin(angle)*v[:,:,2] for v in P];lo=min(v[:,:,1].min() for v in P);hi=max(v[:,:,1].max() for v in P);mid=(hi+lo)/2;scale=min(400/max(hi-lo,1e-6),350/max(2*np.max(abs(v)) for v in X))*.92;movie=out/f'case{k}.mp4';log=(out/f'case{k}_ffmpeg.log').open('wb');cmd=['ffmpeg','-hide_banner','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','pipe:0','-i',str(source/'viewer'/f'case{k}.wav'),'-c:v','libx264','-preset','fast','-crf','20','-threads','2','-c:a','aac','-b:a','160k','-movflags','+faststart','-shortest',str(movie)];proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
  try:
   for frame in range(CFG['frames']):
    img=Image.new('RGB',(1280,720),(244,241,232));g=ImageDraw.Draw(img);g.text((20,15),'R59 | '+r['dataset']+' | '+r['role']+' | '+r['source_clock_scope'],font=small,fill='#173e36');g.text((20,45),f'frame {frame}/149 | {frame/30:.3f}s | pool {r["pool_size"]} | fixed metadata selection',font=font,fill='#173e36')
    for i,label in enumerate(['Recorded paired source','Heuristic top1','Frozen R41 top1']):
     left=12+426*i;g.rounded_rectangle((left,90,left+402,610),radius=12,fill='white');g.text((left+10,105),label,font=font,fill='#173e36');xy=np.stack((left+201+scale*X[i][frame],350-scale*(P[i][frame,:,1]-mid)),axis=-1)
     for j in range(1,22):g.line([tuple(xy[int(parents[j])]),tuple(xy[j])],fill='#1f8070' if j%2 else '#b17b35',width=4)
    g.text((20,645),'Actual recorded motions, not new generation or artistic ground truth. Same audio/camera/scale.',font=small,fill='#173e36');g.text((20,675),'Body22 root-relative display; fitted/raw source translation retained; Native24 NOT EVALUATED.',font=small,fill='#173e36');proc.stdin.write(img.tobytes())
    if frame==75:img.save(out/f'case{k}.jpg',quality=92)
   proc.stdin.close();assert proc.wait()==0
  except BaseException:proc.terminate();proc.wait();raise
  finally:log.close()
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(movie)]));v=next(s for s in info['streams'] if s['codec_type']=='video');assert int(v['nb_frames'])==CFG['frames'];movies.append(dict(path=movie.name,sha256=sha(movie),frames=CFG['frames'],source_case=r))
 write(out/'result.json',dict(records=movies,generated=False));(out/'index.html').write_bytes(('<meta charset="utf-8"><h1>R59真实配对来源与检索</h1>'+''.join('<p>'+str(i)+' · '+r['source_case']['dataset']+'</p><video controls width="1100" src="'+r['path']+'"></video>' for i,r in enumerate(movies))).encode())
