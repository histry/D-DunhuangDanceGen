"""Source-pose anchored synthetic diagnostics, never real artistic quality labels."""
import json,shutil,hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from dunhuang.research.local_timing_r68.common import read,write,sha
from dunhuang.research.sdmr_warp_rank import connectivity
from dunhuang.research.complete_r36.core import relative_fk
from dunhuang.research.paired_event_r41.events import profiles
from dunhuang.research.omni_judge_r71.prepare import render

ROOT=Path('outputs/judge_repair_r72_01');HUMAN=Path('outputs/human_r72_01');PARENT=Path('outputs/retiming_baselines_r69_01/main')

def synth(pose,alpha):
    R=np.broadcast_to(pose,(240,22,3,3)).copy()
    for joint,amplitude in [(16,.9),(17,-.9),(18,.5),(19,-.5)]:
        v=np.zeros((240,3));v[:,2]=amplitude*alpha;R[:,joint]=pose[joint]@Rotation.from_rotvec(v).as_matrix()
    return R

def prepare():
    out=ROOT/'media';out.mkdir(parents=True,exist_ok=False);ctx=connectivity.prepare();cases=read(PARENT/'population.json')['cases']
    source_ids=sorted({c['source_uid'] for c in cases})[:6];assert len(source_ids)==6
    registration=dict(source_ids=source_ids,development_sources=source_ids[:1],validation_sources=source_ids[1:],
        source_split='RECORDING_DISJOINT_WITHIN_R72; ALL_HISTORICALLY_EXPOSED_TO_PROJECT',
        max_generation_attempts=48,planned_primary_calls=45,planned_replay_calls=1,
        primary_validation_fps=2,development_fps=[1,2],validation_1fps_contrast='SYNC_AB_ONLY',
        model='FROZEN_R71_OFFICIAL_AWQ',prompt_revisions_allowed=0,human_reviewed=False,
        diagnostic_truth='SYNTHETIC_CONSTRUCTION_NOT_DUNHUANG_ARTISTIC_QUALITY',Native24='NOT_EVALUABLE')
    write(ROOT/'registration.json',registration);jobs=[];assets={};descriptions=[]
    for source_index,uid in enumerate(source_ids):
        c=next(c for c in cases if c['source_uid']==uid);a,b=c['source_interval'];pose=ctx['lib'].rotations[uid][a]
        alpha=(1-np.cos(2*np.pi*np.arange(240)/60))/2;moving=synth(pose,alpha);still=synth(pose,np.zeros(240))
        delayed=np.concatenate([moving[:1].repeat(15,axis=0),moving[:-15]])
        localized=synth(pose,np.exp(-((np.arange(240)-120)/9)**2))
        peaks=find_peaks(profiles(moving)['upper'],distance=20)[0];peaks=peaks[(peaks>=15)&(peaks<225)]
        audio=np.zeros(128000,dtype=np.float32)
        for frame in peaks:
            n=800;start=int(frame/30*16000);t=np.arange(n)/16000;audio[start:start+n]+=(.7*np.sin(2*np.pi*660*t)*np.hanning(n)).astype(np.float32)
        wave=out/f'source_{source_index}.wav';sf.write(wave,audio,16000,subtype='PCM_16')
        clips={}
        for name,R in dict(moving=moving,still=still,delayed=delayed,localized=localized).items():
            path=out/f'source_{source_index}_{name}.mp4';render(relative_fk(R,ctx['J'],ctx['parents']),ctx['parents'],path);clips[name]=path
        import subprocess
        for test,left,right in [('Identical','moving','moving'),('Different','moving','still'),('SyncAB','moving','delayed'),('SyncBA','delayed','moving'),('Muted','moving','delayed'),('Localization','localized','localized')]:
            token=hashlib.sha256((uid+test).encode()).hexdigest()[:24];video=out/(token+'.mp4')
            subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(clips[left]),'-i',str(clips[right]),'-i',str(wave),'-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-map','2:a:0','-t','8','-map_metadata','-1','-c:v','libx264','-preset','fast','-crf','20','-c:a','aac',str(video)],check=True)
            role='dev' if source_index==0 else 'validation'
            if role=='dev':rates=[1,2] if test!='Localization' else []
            else:rates=[1,2] if test=='SyncAB' else [2]
            for fps in rates:
                task='naturalness' if test in ['Identical','Different'] else 'time' if test=='Localization' else 'music'
                expected='TIE' if test=='Identical' else 'DIFFERENT_ONLY_NO_NATURALNESS_TRUTH' if test=='Different' else 'UNSURE' if test=='Muted' else 'B' if test=='SyncBA' else 4.0 if test=='Localization' else 'A'
                jobs.append(dict(job_id=f's{source_index}_{test}_{fps}',source_uid=uid,source_index=source_index,source_interval=[a,b],role=role,test=test,task=task,fps=fps,
                    video=str(video),video_sha256=sha(video),audio=str(wave),audio_sha256=sha(wave),use_audio=test not in ['Muted','Localization'],expected=expected,
                    observation_interval_seconds=[.5,8] if test in ['SyncAB','SyncBA','Muted'] else [0,8],
                    support=dict(delayed_source_valid_frames=[15,240],padding_frames=[0,15],circular_shift=False),synthetic_truth=True))
            descriptions.append(dict(source_uid=uid,role=role,test=test,video=video.relative_to(ROOT).as_posix(),expected=expected if rates else None))
    assert len(jobs)==45
    write(ROOT/'jobs.json',jobs);write(ROOT/'construction.json',dict(descriptions=descriptions,synthetic=True,frames=240,fps=30,padding_not_scored=[0,.5],style_truth='UNKNOWN'))
    options=''.join(f'<option value="{p["video"]}">{p["source_uid"]} · {p["role"]} · {p["test"]}</option>' for p in descriptions)
    (ROOT/'index.html').write_bytes(('''<!doctype html><meta charset="utf-8"><title>R72 已知正负对照</title><style>body{background:#102132;color:#eee;font:17px sans-serif;margin:22px}video{width:min(900px,100%)}</style><h1>R72 合成诊断 · 非艺术质量真值</h1><p>ChangE原始姿态上施加已知周期旋转；相同、明显运动差异、单侧延迟15帧及4秒构形峰。用于测试感知与时钟，不能证明真实敦煌舞质量。</p><select id="pick">'''+options+'''</select><br><video id="v" controls></video><p>左右同镜头；延迟分支前15帧为填充，不计同步。Muted项目的模型输入不含音频，视频可由播放器手动静音。模型预登记46次推理（含1次新进程复验），硬上限48次；无自动重试或提示词迭代。</p><script>const p=document.getElementById('pick'),v=document.getElementById('v');p.onchange=()=>{v.pause();v.src=p.value};p.onchange();</script>''').encode('utf-8'))
    write(ROOT/'delivery.json',dict(files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in [ROOT/'index.html',ROOT/'registration.json',ROOT/'construction.json']+[ROOT/x['video'] for x in descriptions]]))
    print('R72_DIAGNOSTICS_PREPARED',len(jobs),flush=True)

def human_benchmark():
    from dunhuang.research.retiming_baselines_r69.viewer import build
    import secrets
    out=HUMAN/'benchmark';out.mkdir(parents=True,exist_ok=False);ctx=connectivity.prepare();cases=read(PARENT/'population.json')['cases'];records=read(PARENT/'records.json');z=np.load(PARENT/'arrays.npz')
    clocks=read('outputs/timing_r64_01/main/music_clocks.json');selected=[];seen=set()
    for song in sorted({c['music_id'] for c in cases}):
        c=next(c for c in cases if c['music_id']==song and (c['source_uid'],tuple(c['source_interval'])) not in seen);selected.append(c);seen.add((c['source_uid'],tuple(c['source_interval'])))
    for c in cases:
        identity=c['source_uid'],tuple(c['source_interval'])
        if identity not in seen:seen.add(identity);selected.append(c)
    assert len(selected)==8 and len({c['music_id'] for c in selected})==3
    display=[]
    for c in selected:
        panels=[]
        for method in ['Uniform','E1W','BeatDP','RhythmDTW','Single64','Multi64']:
            row=next(r for r in records if r['case_key']==c['key'] and r['method']==method);prefix=c['key']+'__'+method+'__'
            panels.append(dict(method=method,row=row,R=z[prefix+'R'] if prefix+'R' in z else None,q=z[prefix+'q'] if prefix+'q' in z else None,full_R=z[prefix+'full_R'] if prefix+'full_R' in z else None))
        a,b=c['source_interval'];display.append((c,panels,dict(R=ctx['lib'].rotations[c['source_uid']][a:b],fps=30)))
    seed=secrets.token_bytes(32);(out/'seed.private').write_bytes(seed);build(out,display,ctx['J'],ctx['parents'],clocks,blind_seed=seed)
    write(out/'review_state.json',dict(source_intervals=8,music_tracks=3,expected_reviewers=[4,6],minimum_independent_people_per_comparison=2,
        real_records=0,state='MISSING',TIE='EQUAL_RATINGS_OR_PAIRWISE_TIE',UNSURE='EXPLICIT_ABSTENTION',MISSING='NO_RECORD_NOT_ZERO',
        existing_importer='dunhuang.research.retiming_baselines_r69.reviews',human_reviewed=False,independent_quality='NOT_EVALUABLE'))
    print('R72_HUMAN_BENCHMARK_READY_0_RECORDS',flush=True)

if __name__=='__main__':
    import sys
    human_benchmark() if '--human' in sys.argv else prepare()
