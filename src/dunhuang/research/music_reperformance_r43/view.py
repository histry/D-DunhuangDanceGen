"""Reuse the tested offline clock/camera with explicit Core permissions."""
import argparse,base64,gzip,json
from pathlib import Path
import numpy as np
import shutil
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.source_context_r42.view import build as parent_view


def build(main,out,controls=None,videos=None):
    main=Path(main);out=Path(out);parent_view(out,extra=main,videos=videos)
    script=(out/'data.js').read_text(encoding='utf8');payload=json.loads(gzip.decompress(base64.b64decode(script.split('"')[1])));data=np.load(main/'arrays.npz');meta=read(main/'result.json');lookup={r['key']:r for r in meta['records']}
    payload['cases']=[c for c in payload['cases'] if c['id'].startswith('new_')]
    for c in payload['cases']:
        key=c['id'][4:];budget=data[key+'_body_budget'];c['permissions']=np.max(budget,axis=-1).tolist();c['phase_anchors']=lookup[key]['phase_anchors'];c['music_main']=c['audio']
    if controls:
        control=Path(controls);z=np.load(control/'arrays.npz');rows={r['key']:r for r in read(control/'result.json')['records']}
        for c in payload['cases']:
            key=c['id'][4:];row=rows[key]
            for method in row['methods']:
                if method=='Geometry':continue
                ids=[]
                for v in z[key+'_'+method+'_22']:
                    name=str(len(payload['tracks']));v=np.asarray(v,dtype='<f4');payload['tracks'][name]=dict(shape=list(v.shape),data=base64.b64encode(v.tobytes()).decode());ids.append(name)
                c['panels'].append(dict(name=method,tracks=ids));c['metrics']['curves'][method]=row['metrics']['curves'][method]
            c['music_alternative']='old_failed.wav' if lookup[key]['target_music']=='new_plan' else 'new_plan.wav'
        for name in ('old_failed.wav','new_plan.wav'):shutil.copyfile(Path('outputs/transition_r37_02/generation')/name,out/name)
    used={k for c in payload['cases'] for p in c['panels'] for k in p['tracks']};payload['tracks']={k:v for k,v in payload['tracks'].items() if k in used}
    encoded=base64.b64encode(gzip.compress(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode(),mtime=0)).decode();(out/'data.js').write_text('window.R42_B64="'+encoded+'";',encoding='utf8')
    js=(out/'viewer.js').read_text(encoding='utf8');js=js.replace("current.panels.some(p=>p.name==='FrozenRTS')?[0,3,5].includes(i):i<4","['真实来源／对齐参考','Geometry','Frozen','Core'].includes(current.panels[i].name)")
    js=js.replace("${frame>=104&&frame<152?'待生成区':'保护／观察区'}","${frame>=104&&frame<152?'Bridge':'Core'} · 该帧最大权限 ${current.permissions[frame].toFixed(3)}rad")
    js=js.replace("$('play').onclick=toggle;","$('musicB').onclick=()=>{audio.pause();playing=false;if(!current.music_alternative)return;current.audio=current.audio===current.music_main?current.music_alternative:current.music_main;audio.src=current.audio;audio.onloadedmetadata=()=>{audio.playbackRate=Number($('rate').value);seek(frame);};};$('play').onclick=toggle;")
    (out/'viewer.js').write_text(js,encoding='utf8');html=(out/'index.html').read_text(encoding='utf8').replace('R42','R43')
    html=html.replace('先看来源，再看模型怎样补间','几何来源与受限Core音乐演绎').replace('参考、几何桥、实际采样与固定 RTS 共用一条时间轴','来源、几何底座和实际生成共用一条时间轴；Core权限独立于Bridge')
    html=html.replace('</main>','<p>阶段代理仅控制编辑权限；未宣称神经网络学会阶段。48帧桥之外的Core有实际非零权限；关键峰值、下肢Core与外端保护。22关节，无手指或接触真值。</p></main>')
    html=html.replace('左：真实隐藏目标，仅供评价；中：冻结R39＋固定RTS；右：主候选＋相同RTS。无音频条目不代表音乐配对。','四栏：真实来源、几何底座、冻结Bridge、R43 Core。Root-relative；不是完整新舞蹈。')
    html=html.replace('48帧有界补间并不等于完整事件音乐重演','Core受限编辑仍不等于完整事件音乐重演')
    html=html.replace('</header>','<button id="musicB">试听另一输入音乐 / 返回原音乐（需已生成对照）</button></header>')
    if 'id="musicB"' not in html:html=html.replace('<main>','<main><button id="musicB">试听另一输入音乐 / 返回原音乐（需已生成对照）</button>')
    (out/'index.html').write_text(html,encoding='utf8')
    write(out/'protocol.json',dict(cases=len(payload['cases']),fps=30,main_arrays_sha256=sha(main/'arrays.npz'),control_arrays_sha256=sha(Path(controls)/'arrays.npz') if controls else None,phase_scope='EDIT_PERMISSION_PROXY_NOT_NEW_NEURAL_PHASE_INPUT',files=[dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.iterdir()) if f.is_file() and f.name!='protocol.json']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--out',required=True);p.add_argument('--controls');p.add_argument('--videos');a=p.parse_args();build(a.main,a.out,a.controls,a.videos)
