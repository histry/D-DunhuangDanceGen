"""Original-clock source previews. Cropped panels never silently time-stretch."""
import base64,json,shutil,subprocess
from .common import *

def preview(out,assets,population):
    out=fresh(out);selected=[]
    for uid in sorted(assets.sources):
        eligible=sorted([u for u in assets.units if u['source_uid']==uid and u['pair']['status']=='MATCHED'],key=lambda u:u['source_interval'])
        if eligible and (assets.sources[uid]['dataset']=='ChangE' or not any(x['dataset']==assets.sources[uid]['dataset'] for x in selected)):
            selected.append(eligible[0])
    cases=[];arrays=dict(J=assets.J,parents=assets.parents)
    for i,u in enumerate(selected):
        pair=u['pair'];lo,hi=u['support_component'];a=min(pair['A4'][0],pair['A5'][0]);b=max(pair['A4'][1],pair['A5'][1])
        L=max(lo,a-CFG['context_frames']);H=min(hi,b+CFG['context_frames']);R=assets.R[u['source_uid']][L:H]
        P=relative_fk(R,assets.J,assets.parents);arrays['case%d_R'%i]=R;arrays['case%d_P'%i]=P
        cuts=[r['source_interval'][1] for r in population['population_views']['legacy499'] if r['source_uid']==u['source_uid'] and L<r['source_interval'][1]<H]
        cases.append(dict(case=i,dataset=u['dataset'],role=assets.sources[u['source_uid']]['role'],source_uid=u['source_uid'],
                          unit_uid=u['unit_uid'],origin=L,frames=H-L,A4=pair['A4'],A5=pair['A5'],pair=pair,
                          legacy_cuts=cuts,protected=assets.sources[u['source_uid']]['protected_spans'],
                          data=base64.b64encode(P.astype('<f4').tobytes()).decode(),
                          selection='FIRST_MATCHED_UNIT_IN_TIME_PER_CHANGE_SOURCE_PLUS_FIRST_SOURCE_PER_PAIRED_DATASET; BEFORE_OUTCOME_SCORES'))
    save_arrays(out/'arrays.npz',arrays);write(out/'cases.json',[{k:v for k,v in c.items() if k!='data'} for c in cases])
    viewer=fresh(out/'viewer');(viewer/'data.js').write_text('window.R60='+json.dumps(dict(cases=cases,parents=assets.parents.tolist()),ensure_ascii=False)+';',encoding='utf8')
    shutil.copyfile(HERE/'viewer.html',viewer/'index.html')
    write(out/'result.json',dict(binding=binding(),cases=len(cases),population_sha256=sha(ROOT/'population/result.json'),
                                arrays_sha256=sha(out/'arrays.npz'),selection_before_outcome_scoring=True,
                                fps=30,original_speed=True,music='SILENT_SOURCE_SEGMENTATION_REVIEW_NOT_MUSIC_MATCHING',
                                source_native_world_root='UNKNOWN_BODY22_ROOT_RELATIVE',generated=False,can_route=False))
    render(out,cases,assets.parents)
    write(out/'viewer/protocol.json',dict(files=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(viewer.iterdir()) if p.is_file()],generated=False))
    return cases

def render(out,cases,parents):
    from PIL import Image,ImageDraw,ImageFont
    movies=fresh(out/'video');fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,18);small=ImageFont.truetype(fontpath,14);rows=[]
    indices=[i for i,c in enumerate(cases) if c['dataset']=='ChangE'][:3]+[i for i,c in enumerate(cases) if c['dataset']!='ChangE']
    for i in indices:
        c=cases[i];P=np.frombuffer(base64.b64decode(c['data']),dtype='<f4').reshape(-1,22,3);X=.94*P[:,:,0]-.342*P[:,:,2];lo=P[:,:,1].min();hi=P[:,:,1].max();mid=(lo+hi)/2;scale=min(370/max(hi-lo,1e-6),350/max(2*np.max(abs(X)),1e-6))*.9
        path=movies/('case%d.mp4'%i);log=(movies/('case%d_ffmpeg.log'%i)).open('wb');proc=subprocess.Popen(['ffmpeg','-hide_banner','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x640','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20','-threads','2','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for f in range(c['frames']):
                actual=c['origin']+f;img=Image.new('RGB',(1280,640),'#f5f1e8');g=ImageDraw.Draw(img)
                g.text((18,12),'R60 source boundary preview | '+c['dataset']+' | original 30 FPS, silent',font=font,fill='#173e36')
                g.text((18,42),'source frame %d | A4 %s | A5 %s | equal length %d'%(actual,c['A4'],c['A5'],c['pair']['length']),font=small,fill='#173e36')
                for k,label in enumerate(('Continuous source + context','A4 R47 protected support','A5 shifted boundary control')):
                    x=12+426*k;g.rounded_rectangle((x,78,x+404,550),radius=12,fill='white');g.text((x+10,90),label,font=font,fill='#173e36')
                    active=k==0 or c['A4' if k==1 else 'A5'][0]<=actual<c['A4' if k==1 else 'A5'][1]
                    color='#177e70' if active else '#d8d8d8';xy=np.stack((x+202+scale*X[f],310-scale*(P[f,:,1]-mid)),axis=-1)
                    for j in range(1,22):g.line([tuple(xy[int(parents[j])]),tuple(xy[j])],fill=color,width=4)
                    if not active:g.text((x+12,510),'Outside candidate: context only',font=small,fill='#8a6660')
                g.text((18,576),'Same original source clock. No resampling, smoothing, neural generation or physical qualification.',font=small,fill='#173e36')
                g.text((18,605),'Protected spans are kinematic proxies; choreographic completeness and world Root/contact remain unknown.',font=small,fill='#173e36');proc.stdin.write(img.tobytes())
                if f==c['frames']//2:img.save(movies/('case%d.jpg'%i),quality=92)
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
            log.close()
        streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)]))['streams'];v=next(x for x in streams if x['codec_type']=='video');assert int(v['nb_frames'])==c['frames']
        rows.append(dict(case=i,path=path.name,sha256=sha(path),frames=c['frames'],silent=True));print('R60_PREVIEW_VIDEO',i,c['frames'],flush=True)
    write(movies/'result.json',dict(records=rows,generated=False))
    (movies/'index.html').write_text('<meta charset="utf-8"><h1>R60 来源边界对照：原速、无音乐、非生成</h1>'+''.join('<p>案例 '+str(r['case'])+'</p><video controls width="1150" src="'+r['path']+'"></video>' for r in rows),encoding='utf8')
