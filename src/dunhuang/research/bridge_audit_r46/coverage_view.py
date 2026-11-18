"""Actual original-source coverage videos; no paired music or generated dance."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from dunhuang.research.complete_r36.core import read,write,sha,Library,relative_fk
from dunhuang.research.reperformance_r44.archive import Arrays
from dunhuang.research.source_context_r42.contracts import exact_arrays
from dunhuang.research.reperformance_r44.replay import digest

def run(out,replay=None):
    out=Path(out)
    if out.exists():raise ValueError('Fresh coverage display')
    out.mkdir(parents=True);parent=Path('outputs/bridge_audit_r46_01');coverage=read(parent/'coverage.json');catalog=read('src/dunhuang/research/complete_r36/config.json')['catalog'];lib=Library(catalog);z=Arrays(parent/'main/arrays.npz');J=z['skeleton_J22'];parents=z['parents'];records=[];arr=dict(skeleton_J22=J,parents=parents);files=[];font=ImageFont.load_default()
    for uid in ('female_FeiTian','female_meditation','male_pipa_2'):
        r=next(r for r in coverage['records'] if r['source_uid']==uid);intervals=[r['current_interval'],r['maximum_mean_activity_256frame_interval']];tracks=[]
        for i,(a,b) in enumerate(intervals):
            R=lib.rotations[uid][a:b];assert len(R)==256;P=relative_fk(R,J,parents);arr[uid+'_'+str(i)+'_R']=R;arr[uid+'_'+str(i)+'_xyz']=P;tracks.append(P)
        records.append(dict(source_uid=uid,intervals=intervals,source_sha256=r['catalog_source_sha256'],current_mean_upper_speed_radps=r['current_mean_upper_speed_radps'],maximum_mean_upper_speed_radps=r['maximum_mean_activity_256frame_upper_speed_radps'],scope='ORIGINAL_NATIVE_RECORDING_WINDOWS; MAX_ACTIVITY_DIAGNOSTIC_NOT_FULL_SEMANTIC_EVENT; NO_MUSIC_NO_GENERATOR'))
        if replay:continue
        file=out/(uid+'.mp4');p=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1000x540','-r','30','-i','-','-c:v','libx264','-preset','veryfast','-crf','22','-threads','2',str(file)],stdin=subprocess.PIPE)
        try:
            for t in range(256):
                image=Image.new('RGB',(1000,540),'#f4f7ef');draw=ImageDraw.Draw(image);draw.text((15,12),'R46 ORIGINAL SOURCE COVERAGE / '+uid,font=font,fill='#234')
                for i,(P,(a,b),label) in enumerate(zip(tracks,intervals,('CURRENT R45 SOURCE WINDOW','MAX MEAN ACTIVITY WINDOW / DIAGNOSTIC'))):
                    draw.text((i*500+15,45),label,font=font,fill='#234');draw.text((i*500+15,70),f'Native source [{a},{b}) / actual frame {a+t}',font=font,fill='#345');pos=[(round(i*500+250+(v[0]*.86-v[2]*.5)*160),round(285-v[1]*160)) for v in P[t]]
                    for j in range(1,22):draw.line([pos[int(parents[j])],pos[j]],fill='#25796e' if j%2 else '#b67c37',width=3)
                draw.text((15,495),'Same camera/scale, common22 Root-relative. Original motion only. No audio pairing or generated motion claimed.',font=font,fill='#345');p.stdin.write(np.asarray(image).tobytes())
        finally:p.stdin.close()
        assert p.wait()==0;assert subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','csv=p=0',str(file)],text=True).strip()=='256';subprocess.run(['ffmpeg','-v','error','-i',str(file),'-f','null','-'],check=True);files.append(dict(name=file.name,bytes=file.stat().st_size,sha256=sha(file),frames=256))
    np.savez_compressed(out/'arrays.npz',**arr);z.close();write(out/'result.json',dict(records=records,coverage_sha256=sha(parent/'coverage.json'),generator_calls=0,optimizer_steps=0))
    if replay:
        exact_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');assert read(out/'result.json')==read(Path(replay)/'result.json');write(out/'receipt.json',dict(exact_source_arrays=True,exact_metadata=True,optimizer_calls=0));return
    html='<!doctype html><meta charset="utf-8"><title>R46 完整来源覆盖对照</title><style>body{max-width:1200px;margin:30px auto;background:#f4f7ef;color:#234;font:18px/1.6 system-ui}video{width:100%}</style><h1>飞天与冥想：当前窗口不代表完整来源</h1><p>左：R45使用的256帧；右：同一真实录制中上身平均角速最高的256帧。固定视角、比例、30FPS。仅比较活动覆盖，不意味着右侧是完整语义原语或更适合音乐。没有生成新动作，也没有给ChangE制造音乐配对；视频无音轨。</p>'
    for r in records:html+='<h2>'+r['source_uid']+'</h2><p>来源区间 '+str(r['intervals'])+'；上身平均角速 '+f"{r['current_mean_upper_speed_radps']:.4f} → {r['maximum_mean_upper_speed_radps']:.4f} rad/s"+'</p><video controls preload="metadata" src="'+r['source_uid']+'.mp4"></video>'
    (out/'index.html').write_text(html,encoding='utf8');files.append(dict(name='index.html',bytes=(out/'index.html').stat().st_size,sha256=sha(out/'index.html')));write(out/'protocol.json',dict(files=files,records=records,scope='SUPPLEMENTAL_SOURCE_COVERAGE_NOT_NEW_GENERATION',source_arrays_sha256=digest(out/'arrays.npz'),coverage_sha256=sha(parent/'coverage.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--replay');a=p.parse_args();run(a.out,a.replay)
