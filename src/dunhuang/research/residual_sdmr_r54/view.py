"""Watch real paired sources; retrieval previews do not pretend to be new dance generation."""
import base64,shutil
import soundfile as sf
import torch
from .common import *
from .learn import Study
from dunhuang.io import checked
from dunhuang.research.strong_baseline_r52.view import finish,encode

def run(out,trained=False):
 out=fresh(out);s=Study();s.frozen_inference=True;net=s.model().eval()
 if trained:net.load_state_dict(torch.load(ROOT/'Conditional/model.pt',map_location=s.device,weights_only=True)['state'])
 tasks=s.meta['tasks'];chosen=[]
 for group in s.meta['groups']['train'][:6]:
  options=[i for i,t in enumerate(tasks) if t['role']=='train' and t['group']==group and t['family']=='cross_music']
  if options:chosen.append(min(options,key=lambda i:(abs(tasks[i]['frames']-150),tasks[i]['uid'])))
 sources={r['source_uid']:r for r in read(Path(CFG['data'])/'manifest.json')['sources']};J=s.z['J'];parents=s.z['parents'];rows=s.meta['rows'];cases=[];folder=out/'viewer';folder.mkdir();results=[]
 for k,i in enumerate(chosen):
  t=tasks[i];r=rows[t['query']];v=s.evaluate(net,t);methods=[('Query source (clock reference)',t['query'])]
  for name in ('Heuristic','Frozen')+ (('Conditional',) if trained else ()):
   score=v[name];best=min(np.flatnonzero(np.isclose(score,score.max(),atol=1e-7,rtol=0)),key=lambda j:rows[t['candidates'][j]]['uid']);methods.append((name,t['candidates'][best]))
  panels=[]
  for name,j in methods:
   row=rows[j];source=sources[row['source_uid']];R=np.load(checked(source['source_arrays']))['R'];a,b=row['source_interval'];P=relative_fk(R[a:b],J,parents).astype('<f4');panels.append(dict(name=name,data=encode(P),jerk=[]))
  source=sources[r['source_uid']];u,vv=r['audio_frame_interval'];path=checked(source['audio']);name='case'+str(k)+'.wav'
  with sf.SoundFile(path) as f:
   sr=f.samplerate;f.seek(round(u*sr/30));wave=f.read(round(vv*sr/30)-round(u*sr/30))
  sf.write(folder/name,wave,sr);meta=dict(scope='R54真实配对录制检索预览，不是新动作生成；时钟关联标签不是艺术适配真值。Train人口先看效果，不使用Val选权重。',task=t,source_rows=[rows[j] for _,j in methods],scores={n:vv.tolist() for n,vv in s.evaluate(net,t).items()},regions=[]);cases.append(dict(label=t['group']+'_'+str(t['frames']),frames=t['frames'],origin=u,audio=name,methods=panels,meta=meta));results.append(dict(task=i,selected=methods))
 finish(folder,cases,parents);page=(folder/'index.html').read_text(encoding='utf8').replace('R52','R54').replace('保留 Heuristic 强基线；生成比较同一完整来源与几何基线，修复评价时间坐标。','真实音乐与真实录制来源的条件困难检索；动作均为来源回放，不是扩散生成。');(folder/'index.html').write_text(page,encoding='utf8');write(folder/'protocol.json',dict(cases=len(cases),generated=False,role='TRAIN_ONLY_FIXED_METADATA_SELECTION',files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.iterdir())]));write(out/'result.json',dict(trained=trained,rows=results,can_route=False))
