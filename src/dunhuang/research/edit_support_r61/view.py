"""Shared absolute source clock; no stretching shorter crops or replacing failed solves."""
import base64,json,subprocess,shutil
from .common import *
def encode(P):return base64.b64encode(np.asarray(P,dtype='<f4').tobytes()).decode()
def page(out,cases,parents,mode):
    folder=fresh(out/'viewer');(folder/'data.js').write_text('window.R61='+json.dumps(dict(cases=cases,parents=parents.tolist(),mode=mode),ensure_ascii=False)+';',encoding='utf8');shutil.copyfile(HERE/'viewer.html',folder/'index.html')
    write(folder/'protocol.json',dict(cases=len(cases),mode=mode,files=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(folder.iterdir()) if p.is_file()],neural_generation=False))
def crop_view(out,assets,cases):
    rows=[]
    for c in cases:
        s=c['selection']
        if s['status']!='MATCHED':continue
        a,b=s['A4'];R=assets.R[c['source_uid']][a:b];P=relative_fk(R,assets.J,assets.parents)
        ev=s['evidence'];x,y=ev['evidence_span'];span=[max(0,ev['frame']-12),min(assets.sources[c['source_uid']]['frames'],ev['frame']+13)] if ev['type']=='speed_peak' else [x,y]
        rows.append(dict(case=c['case'],label=c['source_uid']+' · '+c['phase_type'],frames=b-a,origin=a,A4=s['A4'],A5=s['A5'],raw=[x,y],band=span,center=ev['frame'],
            methods=[dict(name='原始连续来源',data=encode(P),interval=[a,b]),dict(name='A4 · 含完整原始证据',data=encode(P),interval=[a,b]),dict(name='A5 · 实际单端截短',data=encode(P),interval=s['A5'])],
            details=c,scope='嵌套裁剪：A5没有新增来源帧，灰色只是被删去的真实上下文；不变速、不冻结末帧。阶段为自动运动学代理，人工确认UNKNOWN。'))
    page(out,rows,assets.parents,'INTERVENTION_VALIDITY');render(out,rows,assets.parents)

def render(out,rows,parents):
    from PIL import Image,ImageDraw,ImageFont
    folder=fresh(out/'video');font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);records=[]
    for row in rows:
        P=np.frombuffer(base64.b64decode(row['methods'][0]['data']),dtype='<f4').reshape(-1,22,3);X=.94*P[:,:,0]-.342*P[:,:,2];mid=(P[:,:,1].min()+P[:,:,1].max())/2;scale=min(370/max(float(np.ptp(P[:,:,1])),1e-8),340/max(float(2*np.max(abs(X))),1e-8))*.9
        path=folder/('case%d.mp4'%row['case']);log=(folder/('case%d.log'%row['case'])).open('wb');p=subprocess.Popen(['ffmpeg','-hide_banner','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x640','-r','30','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for f in range(row['frames']):
                t=row['origin']+f;im=Image.new('RGB',(1280,640),'#f4f1e8');g=ImageDraw.Draw(im);g.text((20,15),'R61 | '+row['label']+' | original 30 FPS | silent source crop',font=font,fill='#173e36');g.text((20,45),'source frame %d | raw evidence %s | cut %d | no added frames'%(t,row['raw'],row['A5'][1]),font=small,fill='#173e36')
                for i,label in enumerate(('Original continuous source','A4: original raw evidence retained','A5: actual crop (gray = deleted)')):
                    x=12+426*i;g.rounded_rectangle((x,82,x+404,544),radius=10,fill='white');g.text((x+9,96),label,font=font,fill='#173e36');xy=np.stack((x+202+scale*X[f],310-scale*(P[f,:,1]-mid)),axis=-1);active=i<2 or t<row['A5'][1]
                    for j in range(1,22):g.line([tuple(xy[int(parents[j])]),tuple(xy[j])],fill='#177e70' if active else '#d2d2d2',width=4)
                    if not active:g.text((x+12,510),'Deleted source context; NOT crop output',font=small,fill='#a14b3e')
                width=1230/max(row['frames'],1)
                for y,span,color in [(565,row['raw'],'#b87938'),(585,row['A5'],'#177e70')]:g.rectangle((20+(span[0]-row['origin'])*width,y,20+(span[1]-row['origin'])*width,y+9),fill=color)
                g.text((20,610),'Kinematic evidence removal candidate, not human-confirmed choreography or neural generation.',font=small,fill='#173e36');p.stdin.write(im.tobytes())
                if f==row['A5'][1]-row['origin']:im.save(folder/('case%d.jpg'%row['case']),quality=92)
            p.stdin.close();assert p.wait()==0
        finally:
            if p.poll() is None:p.terminate();p.wait()
            log.close()
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)]));assert int(next(v for v in info['streams'] if v['codec_type']=='video')['nb_frames'])==row['frames']
        records.append(dict(case=row['case'],path=path.name,frames=row['frames'],sha256=sha(path)));print('R61_CROP_VIDEO',row['case'],row['frames'],flush=True)
    write(folder/'result.json',dict(records=records,silent=True,generated=False));(folder/'index.html').write_text('<meta charset="utf-8"><h1>R61 原速来源裁剪，灰色仅表示已删除上下文</h1>'+''.join('<p>'+str(r['case'])+'</p><video controls width="1150" src="'+r['path']+'"></video>' for r in records),encoding='utf8')
