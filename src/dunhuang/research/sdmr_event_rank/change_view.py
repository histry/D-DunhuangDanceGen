"""Unpaired ChangE ranking preview. The old selected event is never a positive label."""
import argparse,base64,gzip,json,math,shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from dunhuang.io import checked
from dunhuang.research.complete_r36.core import read,write,sha
from .scorer import partial_score

def encode_track(values,frames):
    p=np.asarray(values,dtype='<f4')
    if p.shape!=(frames,22,3) or not np.isfinite(p).all():raise ValueError('Real full-clock FK coordinates required, never token clock positions')
    return dict(shape=list(p.shape),data=base64.b64encode(p.tobytes()).decode())

def build(shadow,out):
    shadow=Path(shadow);out=Path(out)
    if out.exists():raise ValueError('Fresh ChangE shadow viewer')
    out.mkdir(parents=True);meta=read(shadow/'result.json');z=np.load(shadow/'arrays.npz');cfg=read(Path(__file__).with_name('config.json'));tracks={};cases=[]
    def track(key):
        tracks[key]=encode_track(z[key],row['frames']);return key
    for row in meta['records']:
        key=row['key'];a,b=row['audio_frame_interval'];file=checked(row['query_audio']);info=sf.info(file);wave,sr=sf.read(file,start=round(a/30*info.samplerate),stop=round(b/30*info.samplerate),always_2d=True,dtype='float32');g=math.gcd(sr,22050);wave=resample_poly(wave.mean(-1),22050//g,sr//g);sf.write(out/(key+'.wav'),wave,22050,subtype='PCM_16');candidates=[]
        for j,c in enumerate(row['candidates']):
            stride=cfg['token_stride'];m=z[key+'_c0_mt'][::stride];d=z[key+f'_c{j}_dt'][::stride];mv=z[key+'_c0_mv'][::stride];dv=z[key+f'_c{j}_dv'][::stride];heat=m@d.T;heat=np.where(mv[:,None]&dv[None,:],heat,np.nan);_,_,path=partial_score(m,d,mv,dv,z[key+f'_c{j}_positions'][::stride],cfg);v=row['variants']['actual'][j];candidates.append(dict(metadata=c,track=track(key+f'_c{j}_xyz'),clock_positive=None,scores=v[:5],coverage=[float((mv&dv).sum()/max(mv.sum(),1)),0.,v[5]],heatmap=[[float(x) if np.isfinite(x) else None for x in line] for line in heat],path=path))
        cases.append(dict(uid=key,role='历史曝光非配对',group=row['song'],frames=row['frames'],pool='E1-W合法完整事件',query=row['query_prior_source'],query_track=track(key+'_c0_xyz'),audio=key+'.wav',candidates=candidates))
    data=dict(cases=cases,tracks=tracks,parents=z['parents'].tolist(),methods=['Global','Diagonal','Offset','Partial','Hybrid'],summary=dict(queries=len(cases),unavailable=meta['unavailable'],scope='NO_SEMANTIC_TRUTH_NO_R@1_NO_NEW_PLANNER_ROUTING'),generated_dance=False,can_route=False)
    (out/'data.js').write_text('window.E2_DATA="'+base64.b64encode(gzip.compress(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode(),mtime=0)).decode()+'";',encoding='utf8')
    for file in ('index.html','style.css','viewer.js'):shutil.copyfile(Path(__file__).parent/file,out/file)
    html=(out/'index.html').read_text(encoding='utf8').replace('音乐查询与真实事件候选','敦煌音乐与完整 ChangE 候选').replace('查询真实配对动作','旧计划来源参考（不是正确答案）').replace('子窗口继承Parent Event，不冒充完整语义动作；','本页保留完整候选区间，但语义完整性未经人工确认；').replace('时钟正负只说明配对录制时间，不代表艺术适合性。','本页没有音乐动作对应真值；旧计划选择不是正例。所有候选经过实际E1-W重定时。');(out/'index.html').write_text(html,encoding='utf8')
    js=(out/'viewer.js').read_text(encoding='utf8').replace("c.clock_positive?'时钟正例（非艺术真值）':'时钟负例（非艺术否定）'","c.clock_positive===null?'UNKNOWN：未标注音乐相容性':c.clock_positive?'时钟正例（非艺术真值）':'时钟负例（非艺术否定）'");(out/'viewer.js').write_text(js,encoding='utf8')
    write(out/'protocol.json',dict(shadow_arrays_sha256=sha(shadow/'arrays.npz'),query_count=len(cases),generated_dance=False,semantic_truth='UNKNOWN',files=[dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.iterdir()) if f.is_file() and f.name!='protocol.json']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shadow',required=True);p.add_argument('--out',required=True);a=p.parse_args();build(a.shadow,a.out)
