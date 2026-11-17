"""Actual arrays, one audio clock, click-to-seek clock maps and fixed camera."""
import argparse,base64,gzip,json,shutil
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from .archive import Arrays

def build(main,out):
    main=Path(main);out=Path(out)
    if out.exists():raise ValueError('Fresh viewer')
    out.mkdir(parents=True);z=Arrays(main/'arrays.npz');meta=read(main/'result.json');tracks={};cases=[]
    def track(x):
        key=str(len(tracks));a=np.asarray(x,dtype='<f4');tracks[key]=dict(shape=list(a.shape),data=base64.b64encode(a.tobytes()).decode());return key
    for r in meta['records']:
        key=r['key'];panels=[dict(name='真实来源／对齐参考',reference=True,tracks=[track(relative_fk(z[key+'_reference_R'],z['skeleton_J22'],z['parents']))]*2)]
        for method in r['methods']:panels.append(dict(name=method,tracks=[track(x) for x in z[key+'_'+method+'_22']]))
        shutil.copyfile(main/r['audio'],out/r['audio'])
        if r.get('alternative_audio'):shutil.copyfile(main/r['alternative_audio'],out/r['alternative_audio'])
        cases.append(dict(id=key,name=key,panels=panels,audio=r['audio'],gap=[104,152],seeds=r['seeds'],scope=r['scope'],metrics=r['metrics'],phase_maps=r['phase_maps'],music=r['music'],clocks={m:z[key+'_'+m+'_clock'].tolist() for m in r['methods']}))
    payload=dict(cases=cases,tracks=tracks,parents=z['parents'].tolist(),fps=30,frames=256,whole_song_generated=False)
    (out/'data.js').write_text('window.R42_B64="'+base64.b64encode(gzip.compress(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode(),mtime=0)).decode()+'";',encoding='utf8')
    from dunhuang.research import source_context_r42
    static=Path(source_context_r42.__file__).parent
    for f in ('index.html','viewer.js','style.css'):shutil.copyfile(static/f,out/f)
    html=(out/'index.html').read_text(encoding='utf8').replace('R42','R44').replace('先看来源，再看模型怎样补间','来源姿态不变，随音乐改变演法时间').replace('参考、几何桥、实际采样与固定 RTS 共用一条时间轴','Geometry / E1-W / C²平滑重定时 / 冻结R43失败样本 · 同一音乐时钟').replace('48帧有界补间并不等于完整事件音乐重演','本轮没有新增神经采样；仅在两侧Core调整时钟，Bridge精确保留').replace('动态误差与缺口时间线','世界FK D3与缺口时间线').replace('</main>','<canvas id="mapping" width="1140" height="160"></canvas><p>时间图：横轴目标帧，纵轴来源帧。橙线为音乐拍点；点击可逐帧定位。种子一／二中的几何输出相同，不是独立重复实验。来源含隐藏评价目标，仅用于对照。</p></main>')
    (out/'index.html').write_text(html,encoding='utf8');js=(out/'viewer.js').read_text(encoding='utf8');js=js.replace("${frame>=104&&frame<152?'待生成区':'保护／观察区'}","${frame>=104&&frame<152?'固定Geometry Bridge':'Core / 保护端'}")
    injection="""const mc=$('mapping'),mx=mc.getContext('2d');mx.clearRect(0,0,1140,160);mx.strokeStyle='#ddd';for(const a of current.music.beats){mx.beginPath();mx.moveTo(a/255*1140,0);mx.lineTo(a/255*1140,160);mx.stroke();}Object.entries(current.clocks).forEach(([name,v],j)=>{mx.strokeStyle=['#aaa','#c86','#287','#967','#a36','#859','#999'][j%7];mx.beginPath();v.forEach((s,i)=>{const xx=i/255*1140,yy=150-s/255*140;i?mx.lineTo(xx,yy):mx.moveTo(xx,yy);});mx.stroke();mx.fillText(name,15+j*145,12);});mx.strokeStyle='#111';mx.beginPath();mx.moveTo(frame/255*1140,0);mx.lineTo(frame/255*1140,160);mx.stroke();mc.onclick=e=>{audio.pause();playing=false;seek((e.clientX-mc.getBoundingClientRect().left)/mc.getBoundingClientRect().width*255);};"""
    js=js.replace("x.stroke();}function toggle()","x.stroke();"+injection+"}function toggle()")
    (out/'viewer.js').write_text(js,encoding='utf8');write(out/'protocol.json',dict(cases=len(cases),arrays_sha256=sha(main/'arrays.npz'),new_neural_samples=0,files=[dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.iterdir()) if f.name!='protocol.json']));z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--out',required=True);a=p.parse_args();build(a.main,a.out)
