"""Real source playback and learned clock readout, never mislabeled as generation."""
import soundfile as sf
from .common import *
from .study import Study
from .evaluate import model,clock_readout
from dunhuang.research.complete_r36.core import relative_fk
from dunhuang.research.strong_baseline_r52.view import finish,encode
from dunhuang.io import checked

def run(out,trained=False):
 out=fresh(out);s=Study();s.frozen_inference=True;net=model(s,'Conditional') if trained else s.model().eval();rows=s.meta['rows'];sources={r['source_uid']:r for r in read('outputs/paired_event_r41_01/data/manifest.json')['sources']};chosen=[]
 for group in s.split['optimization'][:6]:
  options=[i for i,t in enumerate(s.meta['tasks']) if t['group']==group and t['family']=='same_music'] or [i for i,t in enumerate(s.meta['tasks']) if t['group']==group];chosen.append(min(options,key=lambda i:(abs(s.meta['tasks'][i]['frames']-150),s.meta['tasks'][i]['uid'])))
 folder=fresh(out/'viewer');cases=[];J=s.z['J'];parents=s.z['parents'];diagnostics=[]
 for k,i in enumerate(chosen):
  t=s.meta['tasks'][i];q=t['query'];r=rows[q];v=s.evaluate(net,t);panels=[];identities=[]
  for name,j in [('Query physical source',q)]+[(name,t['candidates'][int(np.argmax(v[name]))]) for name in ('Heuristic','Frozen')+(('Conditional',) if trained else ())]:
   row=rows[j];src=sources[row['source_uid']];a,b=row['source_interval'];R=np.load(checked(src['source_arrays']))['R'];P=relative_fk(R[a:b],J,parents);panels.append(dict(name=name,data=encode(P),jerk=[]));identities.append(dict(method=name,row=j,source_uid=row['source_uid'],interval=[a,b],event_uid=row['parent_event_uid']))
  src=sources[r['source_uid']];u,end=r['audio_frame_interval'];audio='case'+str(k)+'.wav'
  with sf.SoundFile(checked(src['audio'])) as f:
   sr=f.samplerate;f.seek(round(u*sr/30));wave=f.read(round(end*sr/30)-round(u*sr/30))
  sf.write(folder/audio,wave,sr);clock=[]
  with torch.no_grad():
   for shift in CFG['offsets']:
    raw=s.clock(net,q,shift);base=s.clock_base(q,shift);score=residual_scores(raw,base,np.zeros(5)).cpu().numpy();clock.append(dict(injected=shift,truth=-shift,predicted=CFG['offsets'][int(np.argmax(score))],scores=score.tolist()))
  details=dict(task_uid=t['uid'],query=q,candidates=t['candidates'],positive=t['positive'],sources=identities,scores={n:x.tolist() for n,x in v.items()},clock=clock,offset_candidates=CFG['offsets'],phase_and_semantic_truth='UNKNOWN',Root_contact='UNKNOWN',role='OPTIMIZATION_TRAIN_FIXED_METADATA_NOT_BEST_CASE',generated=False)
  cases.append(dict(label=t['group']+' · '+t['family'],frames=t['frames'],origin=u,audio=audio,methods=panels,meta=dict(scope='R55 真实配对来源回放＋局部时钟恢复；不是扩散生成，不是敦煌艺术适配真值。',regions=[],details=details)));diagnostics.append(details)
 finish(folder,cases,parents)
 page=(folder/'index.html').read_text(encoding='utf8').replace('R52','R55').replace('保留 Heuristic 强基线；生成比较同一完整来源与几何基线，修复评价时间坐标。','真实配对事件选择与已知偏移恢复；页面四栏均为来源动作回放，不是新舞蹈生成。').replace('来源阶段为观测运动学代理，世界 Root／接触及 Native24 尚未取得完整资格。','主候选先展示，后续同预算对照另行执行。')
 page=page.replace('</style>','#curve{display:none} #alignment{background:white;padding:12px;margin:12px 0} table{width:100%;border-collapse:collapse} td,th{padding:6px;border-bottom:1px solid #ddd}</style>').replace('相对身体 D3 诊断曲线（不是 Native Jerk；点击跳转）。高值须结合来源活动与动作阶段判断。','时钟恢复诊断：注入偏移只代表受控物理时间变化；不能证明艺术语义。').replace('<details><summary>','<section id="alignment"></section><details><summary>')
 page=page.replace('</script></html>',r"""const originalShow=show;show=function(t){originalShow(t);const d=row.meta.details;const head='<h2>已知非循环偏移：真实标签与主候选预测（帧）</h2><table><tr><th>注入动作偏移</th><th>恢复目标</th><th>预测</th><th>局部残差分数</th></tr>';document.getElementById('alignment').innerHTML=head+d.clock.map(x=>'<tr><td>'+x.injected+'</td><td>'+x.truth+'</td><td>'+x.predicted+'</td><td>'+x.scores.map(v=>v.toFixed(4)).join(' / ')+'</td></tr>').join('')+'</table><p>候选偏移 '+d.offset_candidates.join(', ')+'。未通过时仍显示真实结果，不挑选成功案例。</p>'};show(frame);
</script></html>""")
 (folder/'index.html').write_text(page,encoding='utf8');write(folder/'protocol.json',dict(cases=len(cases),generated=False,trained=trained,files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.iterdir()) if p.name!='protocol.json']));write(out/'result.json',dict(trained=trained,rows=diagnostics,can_route=False))
