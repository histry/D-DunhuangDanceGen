"""A small synchronized Core comparison using the R65 single-music-clock contract."""
import json,subprocess
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import relative_fk
from .common import write,sha

def build(target,display,J,parents,music):
    watch=Path(target)/'viewer';watch.mkdir()
    items=[];audio_files={}
    for case,results in display:
        A,B=case['target_interval'];clock=music[str(case['songidx'])]
        audio_key=case['audio_sha256'][:12]+'_'+str(A)+'_'+str(B)
        name=audio_key+'.wav'
        if name not in audio_files:
            subprocess.run(['ffmpeg','-nostdin','-v','error','-i',clock['audio_path'],'-ss',str(A/30),'-t','8',
                '-vn','-c:a','pcm_s16le','-ac','1','-ar','22050',str(watch/name)],check=True)
            audio_files[name]=True
        panels=[]
        for result in results:
            row=result['row']
            if result['R'] is None:
                panels.append(dict(method=result['method'],status=row['status'],reason=row['warp'].get('reason')))
                continue
            measurement=row['metrics']
            panels.append(dict(method=result['method'],status='LEGAL',P=np.round(relative_fk(result['R'],J,parents),5).tolist(),
                q=np.round(result['q'],6).tolist(),D3=measurement['actual']['FK']['D3']['curve'],
                phase_target_frames=[p['target_frame'] for p in measurement['source_phase_diagnostics'] if p['type'].endswith('speed_peak')],
                diagnostics=dict(actual_music_proxy=row['actual_music_reward']['primary'],
                    FK=measurement['actual']['FK'],endpoint_speed=measurement['actual']['endpoint_mean_joint_speed_radps'],
                    range_ratios=measurement['versus_uniform'],sampler=measurement['reconstruction_check'],
                    actual_output_peaks=measurement['actual_output_peaks']['status'],
                    rate_proof=row['warp'].get('proof'),candidate_counts=row['warp'].get('candidate_counts'))))
        items.append(dict(key=case['key'],source=case['source_uid'],source_interval=case['source_interval'],
            target_interval=[A,B],music=case['music_id'],audio=name,panels=panels,
            beats=[t-A for t in clock['beat_frames'] if A<=t<B],
            accents=[t-A for t in clock['accent_frames'] if A<=t<B],
            missing=[r['method'] for r in results if r['R'] is None]))
    data=json.dumps(dict(items=items,parents=list(map(int,parents))),ensure_ascii=False,separators=(',',':'))
    text=Path(__file__).with_name('viewer.html').read_text(encoding='utf8').replace('__DATA__',data)
    (watch/'index.html').write_text(text,encoding='utf8')
    write(watch/'protocol.json',dict(scope='ACTUAL_SOURCE_RESAMPLED_BODY22_CORES_ONLY; SAME_240_FRAME_BUDGET',fps=30,
        camera='SHARED_FIXED_SCALE_ROOT_RELATIVE_Y_UP; NO_PER_PANEL_RESCALE',source_intervals=8,display_positions=len(items),
        labels=dict(Uniform='匀速重定时，非原速',E1W='冻结 E1-W',Rhythm='实验性平滑内部 Warp',ShuffledRhythm='错位音乐代理控制'),
        audio='SHARED_ACTUAL_MUSIC_PCM_MONO22050; SINGLE_MASTER_CLOCK',transition_frames=48,
        transition_regenerated=False,adjacent_bridge_qualification_inherited=False,world_root_contact='UNKNOWN',
        Native24='NOT_EVALUABLE',human_reviewed=False,semantic_stage_truth='UNKNOWN',artistic_quality='NOT_EVALUABLE'))
    write(Path(target)/'viewer_manifest.json',dict(files=[dict(path=str(p.relative_to(target)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(watch.glob('*'))],
        raw_rotations=False,full_audio=False,weights=False,selection_before_scores=True))
