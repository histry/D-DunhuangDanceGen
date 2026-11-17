"""Real query/candidate media on a shared clock; partial paths do not retime playback."""
import argparse,base64,gzip,json,math,shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from dunhuang.io import checked
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from .scorer import partial_score


def build(benchmark,out,features=None,evaluation=None):
    benchmark=Path(benchmark);out=Path(out)
    if out.exists():raise ValueError('Fresh ranking viewer')
    out.mkdir(parents=True);meta=read(benchmark/'manifest.json');cfg=meta['config'];parents=read(Path(cfg['data'])/'manifest.json');sources={s['source_uid']:s for s in parents['sources']};sk=np.load(Path(cfg['data'])/'skeleton.npz');rows=meta['rows'];selected={}
    for k,t in enumerate(meta['tasks']):
        key=t['role'],t['source_group'];choice=(t['pool']!='energy_matched',abs(t['frames']-150),rows[t['query']]['uid'])
        if key not in selected or choice<selected[key][0]:selected[key]=choice,k
    z=np.load(Path(features)/'arrays.npz') if features else None;ev=np.load(Path(evaluation)/'arrays.npz') if evaluation else None;tracks={};source_cache={};result=[]
    def track(i):
        name=str(i)
        if name not in tracks:
            r=rows[i];s=sources[r['source_uid']]
            if r['source_uid'] not in source_cache:source_cache[r['source_uid']]=np.load(checked(s['source_arrays']))['R']
            a,b=r['source_interval'];v=relative_fk(source_cache[r['source_uid']][a:b],sk['J'],sk['parents']).astype('<f4');tracks[name]=dict(shape=list(v.shape),data=base64.b64encode(v.tobytes()).decode())
        return name
    for (_,k) in sorted(selected.values(),key=lambda x:x[1]):
        task=meta['tasks'][k];q=rows[task['query']];s=sources[q['source_uid']];audio=checked(s['audio']);info=sf.info(audio);a,b=[round(t/30*info.samplerate) for t in q['audio_frame_interval']];wave,sr=sf.read(audio,start=a,stop=b,dtype='float32',always_2d=True);wave=wave.mean(-1);g=math.gcd(sr,22050);wave=resample_poly(wave,22050//g,sr//g);audio_name=task['uid']+'.wav';sf.write(out/audio_name,wave,22050,subtype='PCM_16');assert abs(len(wave)/22050-q['frames']/30)<1/22050+1e-5
        candidates=[]
        for j,i in enumerate(task['candidates']):
            values=ev[f'actual_{k}_scores'][j].tolist() if ev is not None else None;coverage=ev[f'actual_{k}_coverage'][j].tolist() if ev is not None else None;heatmap=None;path=None
            if z is not None:
                stride=cfg['token_stride'];m=z[f'a{task["query"]}_mt'][::stride];d=z[f'a{i}_dt'][::stride];mv=z[f'a{task["query"]}_mv'][::stride];dv=z[f'a{i}_dv'][::stride];v=m@d.T;heatmap=np.where(mv[:,None]&dv[None,:],v,np.nan);heatmap=[[float(x) if np.isfinite(x) else None for x in row] for row in heatmap];_,_,path=partial_score(m,d,mv,dv,z[f'a{i}_positions'][::stride],cfg)
            candidates.append(dict(index=i,metadata=rows[i],track=track(i),clock_positive=task['positive'][j],scores=values,coverage=coverage,heatmap=heatmap,path=path))
        result.append(dict(task_index=k,uid=task['uid'],role=task['role'],group=task['source_group'],frames=q['frames'],pool=task['pool'],query=q,query_track=track(task['query']),audio=audio_name,audio_sha256=sha(out/audio_name),candidates=candidates))
    value=dict(cases=result,tracks=tracks,parents=sk['parents'].tolist(),methods=['Duration','Energy','Global','Diagonal','Offset','Partial','Hybrid'],summary=dict(subwindows=len(rows),queries=len(meta['queries']),tasks=len(meta['tasks']),unavailable=len(meta['unavailable']),scope='CLOCK_CORRESPONDENCE_NOT_ARTISTIC_COMPATIBILITY'),generated_dance=False,can_route=False)
    (out/'data.js').write_text('window.E2_DATA="'+base64.b64encode(gzip.compress(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode(),mtime=0)).decode()+'";',encoding='utf8')
    for f in ('index.html','viewer.js','style.css'):shutil.copyfile(Path(__file__).parent/f,out/f)
    write(out/'protocol.json',dict(benchmark_sha256=sha(benchmark/'manifest.json'),queries=len(result),generated_dance=False,can_route=False,files=[dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.iterdir()) if f.is_file() and f.name!='protocol.json']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--benchmark',required=True);p.add_argument('--out',required=True);p.add_argument('--features');p.add_argument('--evaluation');a=p.parse_args();build(a.benchmark,a.out,a.features,a.evaluation)
