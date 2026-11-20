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


if __name__ == '__main__':
    run()
