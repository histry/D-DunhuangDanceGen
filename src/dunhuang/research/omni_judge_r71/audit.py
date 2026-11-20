"""Fresh-process AV admission and full-30-FPS motion replay; no judge opinions."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from .prepare import ROOT, SOURCE, sha, read, write


def run():
    from dunhuang.research.sdmr_warp_rank import connectivity
    from dunhuang.research.local_timing_r68.study import pair_metrics
    from dunhuang.research.warp_coupling_r53.common import array_hash
    ctx = connectivity.prepare()
    arrays = np.load(SOURCE / 'arrays.npz')
    measured = []
    differences = []
    for row in read(SOURCE / 'records.json'):
        if row['status'] != 'LEGAL':
            continue
        R = arrays[row['case_key'] + '__' + row['method'] + '__full_R']
        assert R.shape == (528, 22, 3, 3) and array_hash(R) == row['full_R_sha256']
        actual = pair_metrics(R, ctx['J'], ctx['parents'])
        if actual != row['pair_dynamics']:
            differences.append(dict(case_key=row['case_key'], method=row['method'], field='pair_dynamics'))
        measured.append(dict(case_key=row['case_key'], method=row['method'], full_R_sha256=array_hash(R), pair_dynamics=actual))
    videos = []
    for job in read(ROOT / 'jobs.json'):
        assert sha(job['video']) == job['sha256']
        meta = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', job['video']]))
        v = next(s for s in meta['streams'] if s['codec_type'] == 'video')
        a = next(s for s in meta['streams'] if s['codec_type'] == 'audio')
        assert int(v['nb_read_frames']) == 528 and v['avg_frame_rate'] == '30/1'
        assert v['width'] == 640 and v['height'] == 360
        assert abs(float(v['duration']) - 17.6) < 1/30 and abs(float(a['duration']) - 17.6) < .05
        videos.append(dict(job=job, video=v, audio=a))
    write(ROOT / 'receipts/independent_motion_media_replay.json', dict(
        fresh_process=True, optimizer_calls=0, geometry_calls=0, trajectories=len(measured),
        motion_differences=differences, videos=videos, motion=measured,
        human_reviewed=False, Native24='NOT_EVALUABLE'))
    assert not differences and len(measured) == 69
    print('R71_INDEPENDENT_MOTION_MEDIA_EXACT', len(measured), len(videos), flush=True)


def audit_controls():
    baseline={(j['pair_id'],j['order']):j for j in read(ROOT/'jobs.json')}
    def video_hash(path):
        return subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0','-c','copy','-f','hash','-hash','sha256','-'],text=True).strip()
    def audio(path):
        raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','-'])
        a=np.frombuffer(raw,dtype='<f4')[:281600]
        assert len(a)==281600 and np.isfinite(a).all()
        return a
    cache={k:(video_hash(j['video']),audio(j['video'])) for k,j in baseline.items()}
    rows=[]
    for j in read(ROOT/'jobs_controls.json'):
        key=j['pair_id'],j['order'];h,ref=cache[key]
        assert sha(j['video'])==j['sha256'] and video_hash(j['video'])==h
        row=dict(pair_id=j['pair_id'],order=j['order'],condition=j['condition'],same_video_bitstream=True)
        if j['condition']=='muted':
            assert j['video']==baseline[key]['video']
            row['audio_policy']='NOT_PASSED_TO_PROCESSOR_OR_GENERATOR'
        elif j['condition']=='shift500ms':
            a=audio(j['video']);corr=float(np.corrcoef(a[8000:],ref[:-8000])[0,1]);silence=float(np.sqrt(np.mean(a[:7200]**2)))
            assert corr>.98 and silence<1e-3
            row.update(non_circular_shift_samples=8000,shift_correlation=corr,leading_450ms_rms=silence)
        else:
            assert j['audio_donor']['raw_audio_sha256']!=j['original_raw_audio_sha256']
            a=audio(j['video']);donor=audio(SOURCE/'viewer'/j['audio_donor']['path']);corr=float(np.corrcoef(a,donor)[0,1])
            assert corr>.98
            row.update(different_recording=True,donor_correlation=corr)
        rows.append(row)
    write(ROOT/'receipts/independent_audio_controls.json',dict(fresh_process=True,rows=rows,human_reviewed=False))
    assert len(rows)==18
    print('R71_AUDIO_CONTROLS_EXACT',len(rows),flush=True)


if __name__ == '__main__':
    audit_controls() if '--controls' in sys.argv else run()
