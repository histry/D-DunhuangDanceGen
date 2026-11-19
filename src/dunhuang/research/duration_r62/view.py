"""Saved-array viewing only; preserve real duration and label missing support."""
import base64,json,shutil,subprocess
import numpy as np
from .common import ROOT,HERE,CFG,fresh,read,write,sha,relative_fk

def encoded(P):return base64.b64encode(np.asarray(P,dtype='<f4').tobytes()).decode()

def panel(name,z,key,J,parents,origin,frames,edit,truth=False):
    return dict(name=name,data=encoded(relative_fk(z[key],J,parents)) if key and key in z else None,
        origin=int(origin),frames=int(frames),edit=list(map(int,edit)),truth=truth)

def page(out,rows,parents):
    folder=fresh(out/'viewer')
    (folder/'data.js').write_text('window.R62='+json.dumps(dict(cases=rows,parents=parents.tolist()),ensure_ascii=False)+';',encoding='utf8')
    shutil.copyfile(HERE/'viewer.html',folder/'index.html')
    write(folder/'protocol.json',dict(cases=len(rows),original_fps=30,source_duration_preserved=True,
        files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.iterdir()) if p.is_file()],
        not_new_neural_generation=True,world_root_contact='UNKNOWN',Native24='NOT_EVALUABLE'))

def build(out,controls=False):
    cont=read(out/'continuous/result.json');cross=read(out/'cross_source/result.json')
    rows=[];movies=[]
    with np.load(out/'continuous/arrays.npz') as z:
        J,parents=z['J'],z['parents']
        for r in cont['records']:
            a,b=r.get('source_interval',[0,r['gap']+8]);edit=r.get('edit_source_interval',[a+4,b-4])
            panels=[panel('真实隐藏动作',z,r.get('truth_arrays_key'),J,parents,a,b-a,edit,truth=True)]
            for name in ('G1','SLERP','SingleChartHermite'):
                if name not in r.get('methods',{}):continue
                panels.append(panel(name,z,r['methods'][name].get('arrays_key'),J,parents,a,b-a,edit))
            rows.append(dict(label=r['source_uid']+' · '+str(r['gap'])+'帧真实缺口',origin=a,frames=b-a,source_clock=True,
                panels=panels,columns=len(panels),scope='同一长度内共享真实隐藏目标。不同长度改变了目标与边缘观测，不能按误差直接排名时长优劣。灰空栏表示不在真实范围／构造失败。',
                details=dict(case=r['case'],status=r['status'],source_uid=r['source_uid'],gap=r['gap'],source_interval=[a,b],edit=edit,
                    methods=r.get('methods'),difficulty=r.get('difficulty'),phase_truth='UNKNOWN')))
        if not controls:
            # Fixed metadata examples, never picked from reconstruction scores.
            for case in (0,4,8):
                rr=[r for r in cont['records'] if r['case']==case]
                if not rr:continue
                start=min(r['source_interval'][0] for r in rr);end=max(r['source_interval'][1] for r in rr);panels=[]
                for name in ('truth','G1'):
                    for r in rr:
                        key=r.get('truth_arrays_key') if name=='truth' else r.get('methods',{}).get('G1',{}).get('arrays_key')
                        panels.append(panel(str(r['gap'])+'f '+name,z,key,J,parents,r['source_interval'][0],r['gap']+8,r['edit_source_interval'],truth=name=='truth'))
                movies.append(dict(label=rr[0]['source_uid']+' | continuous gaps; original source clock',origin=start,frames=end-start,
                    panels=panels,columns=5,source_clock=True,scope='Same-source targets change with gap; NOT duration quality ranking'))
        page(out/'continuous',rows,parents)
    rows=[]
    with np.load(out/'cross_source/arrays.npz') as z:
        J,parents=z['J'],z['parents']
        for case in sorted({r['pair_case'] for r in cross['records']}):
            rr=[r for r in cross['records'] if r['pair_case']==case];panels=[]
            for name in (('G1','G0') if controls else ('G1',)):
                for r in rr:
                    g=r['gap_frames'];panels.append(panel(str(g)+'帧 · '+name,z,r['methods'][name].get('arrays_key'),J,parents,0,r['frames'],r['edit_target_interval']))
            row=dict(label='pair '+str(case)+' · '+rr[0]['source_uid'],origin=0,frames=max(r['frames'] for r in rr),source_clock=False,
                panels=panels,columns=5,scope='来源Core／Warp／边缘姿态完全固定，原速播放，只改变插入时间。更长过渡增加片段总时长，不是固定音乐预算收益；中间没有唯一真实动作。',
                details=dict(pair_case=case,left=rr[0]['left'],right=rr[0]['right'],durations=[dict(gap=r['gap_frames'],methods=r['methods']) for r in rr],
                    clocks='LEFT_AND_RIGHT_HAVE_DIFFERENT_SOURCE_CLOCKS; GAP_HAS_NO_SOURCE_TRUTH',human_reviewed=False))
            rows.append(row)
            if not controls and case==0:
                movies.append(dict(row,label='cross pair 0 | fixed cores, extra transition time',scope='Cross-source: no hidden truth; longer gap adds time'))
        page(out/'cross_source',rows,parents)
    if not controls:render(out,movies,parents)
    controls_link='<a href="controls/continuous/viewer/index.html">④ 同长度三方法对照</a><a href="controls/cross_source/viewer/index.html">⑤ 固定Core的G0／G1对照</a>' if controls else ''
    (ROOT/'index.html').write_text('<meta charset="utf-8"><style>body{max-width:1100px;margin:45px auto;font:20px system-ui;background:#f4f1e8;color:#173e36}a{display:block;margin:24px}</style><h1>R62 · 过渡时长诊断</h1><a href="main/continuous/viewer/index.html">① 12条真实来源 × 五种缺口：真实动作／G1</a><a href="main/cross_source/viewer/index.html">② 固定来源Core：五种过渡时长并排</a><a href="main/video/index.html">③ 四段预登记原速时长对照视频</a>'+controls_link+'<p>先观看主效果，再运行对照。连续重建和跨来源连接分开；身体22关节Root-relative，阶段语义／世界Root／接触未知。没有新训练、没有联合规划升级。</p>',encoding='utf8')

def render(out,rows,parents):
    from PIL import Image,ImageDraw,ImageFont
    folder=fresh(out/'video');font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13);records=[]
    for num,row in enumerate(rows):
        arrays=[np.frombuffer(base64.b64decode(p['data']),dtype='<f4').reshape(-1,22,3) if p['data'] else None for p in row['panels']]
        valid=[p for p in arrays if p is not None];allP=np.concatenate(valid);lo,hi=float(allP[:,:,1].min()),float(allP[:,:,1].max());xx=float(np.max(abs(.94*allP[:,:,0]-.342*allP[:,:,2])))
        cols=5;lines=(len(arrays)+cols-1)//cols;W,H=1500,100+lines*350+70
        path=folder/('example%d.mp4'%num);log=(folder/('example%d.log'%num)).open('wb')
        proc=subprocess.Popen(['ffmpeg','-hide_banner','-y','-f','rawvideo','-pix_fmt','rgb24','-s',str(W)+'x'+str(H),'-r','30','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for f in range(row['frames']):
                t=row['origin']+f;im=Image.new('RGB',(W,H),'#f4f1e8');g=ImageDraw.Draw(im)
                g.text((15,12),'R62 | '+row['label'],font=font,fill='#173e36');g.text((15,40),'30 FPS | silent | frame %d | %s'%(t,row['scope']),font=small,fill='#173e36')
                for i,(p,P) in enumerate(zip(row['panels'],arrays)):
                    x,y=(i%cols)*300+8,90+(i//cols)*350;g.rounded_rectangle((x,y,x+284,y+338),radius=8,fill='white');g.text((x+10,y+8),p['name'],font=font,fill='#173e36');idx=t-p['origin']
                    if P is None or not 0<=idx<len(P):
                        g.text((x+12,y+135),'No actual frame / rejected',font=small,fill='#8b5941');continue
                    s=min(250/(hi-lo or 1),250/(2*xx or 1))*.9;px=P[idx];xy=np.stack((x+142+s*(.94*px[:,0]-.342*px[:,2]),y+170-s*(px[:,1]-(hi+lo)/2)),axis=-1)
                    for j in range(1,22):g.line([tuple(xy[int(parents[j])]),tuple(xy[j])],fill='#728376' if p['truth'] else '#177e70',width=3)
                    g.text((x+10,y+311),'edit' if p['edit'][0]<=t<p['edit'][1] else 'real boundary',font=small,fill='#a46c36')
                g.text((15,H-45),'Body22 root-relative; unknown world Root/contact. Not Native24 or perceptual quality qualification.',font=small,fill='#173e36');proc.stdin.write(im.tobytes())
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
            log.close()
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)]));assert int(next(s for s in info['streams'] if s['codec_type']=='video')['nb_frames'])==row['frames']
        records.append(dict(path=path.name,label=row['label'],frames=row['frames'],sha256=sha(path)));print('R62_VIDEO',num,row['frames'],flush=True)
    write(folder/'result.json',dict(records=records,silent=True,not_neural_generation=True))
    (folder/'index.html').write_text('<meta charset="utf-8"><h1>R62 原速时长对照：空栏不是补帧</h1>'+''.join('<p>'+r['label']+'</p><video controls width="1350" src="'+r['path']+'"></video>' for r in records),encoding='utf8')
