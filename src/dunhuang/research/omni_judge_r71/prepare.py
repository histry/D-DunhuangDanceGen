"""Frozen R69 AV judge media. No new motion, human labels, or training."""
import json,hashlib,subprocess
from pathlib import Path
import numpy as np

ROOT=Path('outputs/omni_judge_r71_01');SOURCE=Path('outputs/retiming_baselines_r69_01/main')
METHODS=['Uniform','BeatDP','RhythmDTW','Single64','Multi64']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_bytes((json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode())
def eligible(row):return row['status']=='LEGAL' and (row.get('connection') or {}).get('passes_original_screen') is True
def register(cases,rows):
 by={(r['case_key'],r['method']):r for r in rows};result=[]
 for i,c in enumerate(cases):
  for method in METHODS:
   valid=eligible(by[c['key'],'E1W']) and eligible(by[c['key'],method]);key=hashlib.sha256((c['key']+method).encode()).hexdigest()[:24]
   result.append(dict(pair_id=key,case_key=c['key'],case_index=i,method=method,eligible=valid,pilot=(i in (0,4,7) and method=='Single64'),left='E1W',right=method,
    exclusion=None if valid else dict(baseline=by[c['key'],'E1W']['status'],candidate=by[c['key'],method]['status'],candidate_G1=(by[c['key'],method].get('connection') or {}).get('passes_original_screen'))))
 return result
def render(P,parents,path):
 from PIL import Image,ImageDraw
 process=subprocess.Popen(['ffmpeg','-v','error','-nostdin','-f','rawvideo','-pix_fmt','rgb24','-s','320x360','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
 for pose in P:
  image=Image.new('RGB',(320,360),(16,33,50));d=ImageDraw.Draw(image);xy=[(160+125*(p[0]+.35*p[2]),180-125*(p[1]-.15*p[2])) for p in pose]
  if not all(5<=x<315 and 5<=y<355 for x,y in xy):raise ValueError('Common camera cropped body; media not admitted')
  for j in range(1,22):d.line([xy[int(parents[j])],xy[j]],fill=(136,220,200),width=3)
  process.stdin.write(image.tobytes())
 process.stdin.close()
 if process.wait():raise RuntimeError('Video encoding failed')
def prepare(scope='pilot'):
 from dunhuang.research.complete_r36.core import relative_fk
 from dunhuang.research.sdmr_warp_rank import connectivity
 root=ROOT/('media_v2' if scope=='pilot' else 'media_full');root.mkdir(parents=True,exist_ok=False)
 assert sha(SOURCE/'manifest.json')=='57ea35f5d7bd1d5d63da065e46e59951698d78b05c844de9aae711f890b08abe'
 manifest=read(SOURCE/'manifest.json')
 for name,h in manifest['files'].items():
  p=(SOURCE/name).resolve();assert p.is_relative_to(SOURCE.resolve()) and sha(p)==h
 cases=read(SOURCE/'population.json')['cases'];rows=read(SOURCE/'records.json');pairs=register(cases,rows)
 assert len(pairs)==60 and sum(r['eligible'] for r in pairs)==56
 write(ROOT/'registered_pairs.json',pairs)
 ctx=connectivity.prepare();z=np.load(SOURCE/'arrays.npz');jobs=[];singles={}
 for pair in pairs:
  if not pair['eligible'] or scope=='pilot' and not pair['pilot']:continue
  c=cases[pair['case_index']];a,b=c['pair_music_interval'];audio=next(SOURCE/'viewer'/p['path'] for p in read(SOURCE/'viewer/protocol.json')['audio_files'].values() if p['pair_music_interval']==[a,b] and p['raw_audio_sha256']==c['audio_sha256'])
  for method in ['E1W',pair['method']]:
   key=(c['key'],method)
   if key not in singles:
    name=hashlib.sha256(str(key).encode()).hexdigest()[:24]+'.mp4';r=z[c['key']+'__'+method+'__full_R'];assert r.shape==(528,22,3,3)
    render(relative_fk(r,ctx['J'],ctx['parents']),ctx['parents'],root/name);singles[key]=root/name
  for order in ['AB','BA']:
   sides=['E1W',pair['method']] if order=='AB' else [pair['method'],'E1W'];out=root/(pair['pair_id']+'_'+order+'.mp4')
   cmd=['ffmpeg','-v','error','-nostdin','-i',str(singles[c['key'],sides[0]]),'-i',str(singles[c['key'],sides[1]]),'-i',str(audio),'-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-map','2:a:0','-t','17.6','-map_metadata','-1','-c:v','libx264','-preset','fast','-crf','20','-c:a','aac',str(out)]
   subprocess.run(cmd,check=True)
   jobs.append(dict(pair_id=pair['pair_id'],order=order,video=str(out),sha256=sha(out),frames=528,fps=30,duration=17.6,condition='correct_audio'))
 write(ROOT/('jobs.json' if scope=='pilot' else 'jobs_full.json'),jobs);write(ROOT/('media_manifest.json' if scope=='pilot' else 'media_manifest_full.json'),dict(source_manifest_sha256=sha(SOURCE/'manifest.json'),scope=scope,jobs=jobs,registered=60,eligible=56,missing_not_replaced=4,human_reviewed=False))
 if scope=='pilot':
  import html
  options=''.join('<option value="'+html.escape(j['video'].replace('outputs/omni_judge_r71_01/',''))+'">案例 '+str(i//2+1)+' · '+j['order']+'</option>' for i,j in enumerate(jobs))
  page='''<!doctype html><meta charset="utf-8"><title>R71 实际音视频评价先导</title><style>body{background:#102132;color:#ddd;font:17px sans-serif;margin:24px}video{width:min(960px,100%)}button,select{padding:8px;margin:6px}small{display:block;color:#ddbd81}</style><h1>R71 · 三组实际 A/B 片段</h1><p>R69 冻结轨迹；每组528帧、30 FPS、17.6秒。左右同音乐、相机及比例，仅交换位置。AB：A=E1-W，B=Single64；BA顺序相反。此页展示真实输入，不是模型评价结果。</p><select id="pick">'''+options+'''</select><br><video id="v" controls preload="metadata"></video><br><button id="prev">前一帧</button><button id="next">后一帧</button><span id="clock"></span><small>身体22关节、Root-relative视角。世界Root、接触、Native24与独立艺术质量不可评。模型将按1 FPS采样，不能据此评价30 FPS细节。自动偏好不计入人工记录。</small><script>const v=document.getElementById('v'),p=document.getElementById('pick');function load(){v.pause();v.src=p.value}p.onchange=load;document.getElementById('prev').onclick=()=>{v.pause();v.currentTime=Math.max(0,(Math.floor(v.currentTime*30+1e-4)-1)/30)};document.getElementById('next').onclick=()=>{v.pause();v.currentTime=Math.min(527/30,(Math.floor(v.currentTime*30+1e-4)+1)/30)};v.ontimeupdate=()=>document.getElementById('clock').textContent=Math.floor(v.currentTime*30+1e-4)+' / 527 · '+v.currentTime.toFixed(3)+'秒';load();</script>'''
  (ROOT/'index.html').write_bytes(page.encode('utf-8'))
 print('R71_MEDIA_READY',len(jobs),flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--scope',choices=['pilot','full'],default='pilot');a=p.parse_args();prepare(a.scope)

