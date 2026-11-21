"""Reuse actual frozen dances for A; metadata-first displays for B/C."""
import json
from pathlib import Path
import numpy as np
from .common import ROOT,OLD,CFG,HERE,read,write,sha,prepare,fresh
from dunhuang.research.complete_r36.core import relative_fk

def payload(path):return json.loads(Path(path).read_text(encoding='utf8').split('<script id="data" type="application/json">',1)[1].split('</script>',1)[0])

def render(out,data,extra=''):
    t=(HERE.parent/'source_time_q01/viewer.html').read_text(encoding='utf8');t=t.replace('<p id="mode"></p>','<p id="mode"></p><canvas id="reach" width="600" height="180" style="width:min(600px,100%)"></canvas><pre id="certificate"></pre>');t=t.replace("function draw(n){let x=D.items[i];", "function draw(n){let x=D.items[i];drawCertificate(x);")
    helper="""function drawCertificate(x){let g=$('reach').getContext('2d');g.clearRect(0,0,600,180);let cert=x.certificates||[];cert.forEach((r,k)=>{let y=25+k*60,b=r.continuous_reachable_interval;g.fillStyle='#809eb4';g.fillText('S'+r.source_order+' 连续速率域',10,y);if(b){g.fillStyle=r.rate_status==='EXCLUDED_BY_RATE'?'#b7784f':'#50ad99';g.fillRect(160+b[0]/(x.core-1)*420,y-12,(b[1]-b[0])/(x.core-1)*420,14)}g.fillStyle='#dd8bbb';g.fillRect(160+(x.target-3)/(x.core-1)*420,y-15,6/(x.core-1)*420,20)});$('reach').hidden=!cert.length;$('certificate').textContent=x.explanation||''} """
    t=t.replace("c.fillRect(x.target/x.frames*300,325,2,28);","if(x.target!=null)c.fillRect(x.target/x.frames*300,325,2,28);if(r.target_marks)for(let v of r.target_marks){c.fillStyle='#dd8bbb';c.fillRect((v-3)/x.frames*300,325,6/x.frames*300,20)}if(r.anchor_marks)for(let v of r.anchor_marks){c.fillStyle='#70e1cb';c.fillRect(v/x.frames*300,328,2,22)}");t=t.replace('function stop(){',helper+'function stop(){');(out/'index.html').write_bytes(t.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':'))).encode())
    write(out/'protocol.json',dict(files={'index.html':dict(bytes=(out/'index.html').stat().st_size,sha256=sha(out/'index.html'))},existing_local_audio=data.get('audio_bindings',[]),body22=True,can_route=False))

def build_A(root):
    out=fresh(root/'viewer');d=payload(OLD/'preview/viewer/index.html');rows=read(root/'certificates.json')
    for item in d['items']:
        if item['raw']:continue
        source,f=item['label'].split(' · 时间目标 ');r=next(r for r in rows if r['source_uid']==source and r['fraction']==float(f));item['certificates']=r['sources'];item['explanation']='仅速率可达上限：'+('允许；实际成功见证已核验' if r['actual_success'] else '两个来源均排除；增加同条件Warp搜索不能满足目标')+'。区间不包含G1／保护边缘的充分性资格。'
    p=read(OLD/'preview/viewer/protocol.json');d['audio_bindings']=[dict(name=a['name'],existing_relative_path='outputs/paper1_source_time_01/preview/viewer/'+a['name'],sha256=a['sha256']) for a in p['reused_audio']];d['intro']='A：真实冻结执行＋精确连续速率证书。区间来自来源锚点、实际端点和速率，不使用GTW/BAS。可达不是物理或艺术质量证明。';render(out,d);print('RATE_CERTIFICATE_VIEW_READY',flush=True)

def build_B():
    root=ROOT/'controls';out=fresh(root/'viewer');ctx=prepare();pop=read(root/'population.json');rows=read(root/'results.json');tracks={};items=[];audio=[]
    with np.load(root/'arrays.npz',allow_pickle=False) as z:
        def track(k):
            if k not in z:return None
            if k not in tracks:tracks[k]=np.round(relative_fk(z[k],ctx['J'],ctx['parents']),4).tolist()
            return k
        for display in pop['display']:
            if display['unit_uid'] is None:continue
            u=next(u for u in pop['units'] if u['unit_uid']==display['unit_uid']);T=u['target_frames'];variants=[dict(label='Uniform Core · 无连接资格',track=None,core_track=track(u['unit_uid']+'__Uniform_R'),qualified=False,source=str(u['source_interval']))]
            for name in CFG['hypotheses']:
                r=next(r for r in rows if r['unit_uid']==u['unit_uid'] and r['hypothesis']==name);key=r['key'];variants.append(dict(label=name+' · '+r['status']+' · G1 '+str(r.get('actual_G1_pass','UNKNOWN')),track=track(key+'__full_R'),core_track=track(key+'__R'),qualified=r.get('actual_G1_pass',False),source=str(u['source_interval']),q=None if key+'__q' not in z else np.round(z[key+'__q'],6).tolist(),source_interval=u['source_interval'],target_marks=[] if any(e is None for e in u['targets'][name]) else [e['local_target_frame'] for e in u['targets'][name]],anchor_marks=[] if key+'__q' not in z else np.interp([e['source_frame'] for e in u['source_events']],z[key+'__q'],np.arange(T)).tolist(),details={k:v for k,v in r.items() if k not in ['dynamics','connection']}))
            name='dunhuangwu'+str(u['music_song']+2)+'_'+str(T)+'.wav';raw=Path('outputs/complete_r36_01/viewer/audio')/('dunhuangwu'+str(u['music_song']+2)+'.wav');audio.append(dict(name=name,source_existing_relative_path=str(raw),source_sha256=sha(raw),start_frame=900,frames=2*T+48,fps=30,local_excerpt_reconstruction=True));items.append(dict(label=u['source_uid']+' · 事前首个完整代理三元组',frames=2*T+48,core=T,target=None,raw=False,audio=name,variants=variants,explanation='Onset／Peak／Low-run-end均为自动运动学代理；有限对应假设只检验技术可控性，无独立语义正确性。Uniform只显示Core，末端无合格连接时不补假动作。'))
            rawk=u['unit_uid']+'__raw_R';items.append(dict(label=u['source_uid']+' · 原速来源',frames=len(z[rawk]),core=0,target=None,raw=True,audio=None,variants=[dict(label='原速来源',track=track(rawk),qualified=None)]))
    data=dict(tracks=tracks,items=items,parents=ctx['parents'].tolist(),audio_bindings=audio,intro='B/C：完整139单位先登记，再计算多阶段可达范围和三种有限对应假设。展示来源由元数据事前确定，不按可达或G1结果挑选；缺失和失败保留。没有质量排名。');render(out,data);print('MULTI_ANCHOR_VIEW_READY',flush=True)

if __name__=='__main__':build_B()
