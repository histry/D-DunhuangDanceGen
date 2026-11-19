"""Actual fixed-budget body22 playback with one master music clock.

The release packet contains display coordinates/audio, never rotation arrays or
private run records. Rejected policies remain blank; no baseline substitutes.
"""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .common import ROOT, HERE, read, write, sha, relative_fk, array_hash, fresh


FPS, FRAMES = 30, 528
MAIN_POLICIES = ('Fixed48', 'ShortestScreened')
CONTROL_POLICIES = MAIN_POLICIES+('Fixed24', 'BoundaryConditioned', 'Fixed48_G0')


def _audio(case, out, seen):
    source = Path(case['audio_path'])
    digest = sha(source)
    if digest != case['audio_sha256']:
        raise ValueError('Music input changed before viewer rendering: '+case['key'])
    A, B = map(int, case['music_interval'])
    if B-A != FRAMES:
        raise ValueError('Viewer requires the exact 528-frame music budget')
    key = hashlib.sha256((digest+':%d:%d' % (A, B)).encode()).hexdigest()[:20]
    if key not in seen:
        path = out/'audio'/(key+'.m4a')
        path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            raise RuntimeError('ffmpeg is required for exact-clock viewer music excerpts')
        # Decode before trimming; avoid seek-to-keyframe or copied audio packets.
        trim = 'atrim=start=%.12f:end=%.12f,asetpts=PTS-STARTPTS' % (A/FPS, B/FPS)
        subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
            '-i', str(source), '-vn', '-af', trim, '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart', str(path)], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        seen[key] = dict(path='audio/'+path.name, sha256=sha(path),
            source_audio_sha256=digest, source_frame_interval=[A, B],
            source_seconds_interval=[A/FPS, B/FPS], target_duration_seconds=FRAMES/FPS,
            trim='DECODED_ATRIM_ABSOLUTE_SOURCE_SECONDS_RESET_PTS_ZERO',
            source_sample_rate_preserved=True)
    return seen[key]


def _source(side):
    if side is None:
        return dict(source_uid='MISSING', event_uid=None, source_interval=None,
                    semantic_completeness='UNKNOWN')
    return {k: side.get(k) for k in ('source_uid', 'event_uid', 'source_interval',
        'support_component', 'source_role', 'semantic_completeness', 'world_root', 'contact')}


def _panel(policy, selection, records, arrays, J, parents):
    panel = dict(name=policy, status='MISSING_SELECTION', accepted=False,
                 data=None, frames=FRAMES, source_clock=None, regions=[], reason=None)
    if selection is None:
        panel['reason'] = 'No saved selection; no replacement was displayed'
        return panel
    rec = records.get(selection.get('record_key'))
    method = selection.get('method', 'G0' if policy == 'Fixed48_G0' else 'G1')
    panel.update(selection=dict(selection), record_key=selection.get('record_key'),
                 method=method, accepted=bool(selection.get('accepted', False)))
    if rec is None:
        panel.update(status='NO_SELECTED_TRAJECTORY', reason='No saved selected record')
        return panel
    m = rec.get('methods', {}).get(method, {})
    nl = rec['allocation']['left_frames']
    g = rec['gap_frames']
    panel.update(gap_frames=g, allocation=rec['allocation'],
        regions=[dict(kind='LEFT_CORE', interval=[0, nl]),
                 dict(kind='TRANSITION', interval=[nl, nl+g]),
                 dict(kind='RIGHT_CORE', interval=[nl+g, FRAMES])],
        source_clock={s: {k: rec[s].get(k) for k in
            ('source_uid', 'source_interval', 'target_interval', 'time_warp',
             'source_positions', 'full_core_R_sha256', 'status')} for s in ('left', 'right')},
        numerical_screen=m.get('passes_original_screen', False),
        metrics=m.get('metrics'), full_dynamics=m.get('full_dynamics'),
        beat_proxy=m.get('beat_proxy'), core_exact=m.get('core_exact', False))
    if not panel['accepted'] or not m.get('constructible', False):
        panel.update(status='REJECTED_BLANK', reason=m.get('reason',
            'Selection failed the frozen numerical gate; no fallback is displayed'))
        return panel
    key = m.get('arrays_key')
    if key not in arrays:
        raise ValueError('Accepted selection lacks its actual trajectory: '+str(key))
    R = arrays[key]
    if R.shape != (FRAMES, 22, 3, 3) or not np.isfinite(R).all():
        raise ValueError('Accepted trajectory violates body22/528 contract')
    if array_hash(R) != m['full_R_sha256']:
        raise ValueError('Saved playback rotation hash mismatch')
    P = relative_fk(R, J, parents)
    floats = np.ascontiguousarray(P, dtype='<f4')
    panel.update(status='ACTUAL_SAVED_TRAJECTORY', arrays_key=key,
                 rotation_sha256=m['full_R_sha256'],
                 coordinates_sha256=array_hash(floats),
                 coordinate_encoding='BASE64_FLOAT32_LITTLE_ENDIAN_FRAMES_JOINTS_XYZ',
                 data=base64.b64encode(floats.tobytes()).decode('ascii'),
                 bounds=[P.min(axis=(0, 1)).tolist(), P.max(axis=(0, 1)).tolist()])
    return panel


def viewer(root=ROOT, controls=False):
    """Export root/{viewer|controls_viewer}; returns its safe packet manifest."""
    root = Path(root)
    study = root/('controls' if controls else 'main')
    result_path, arrays_path = study/'result.json', study/'arrays.npz'
    result = read(result_path)
    policies = CONTROL_POLICIES if controls else MAIN_POLICIES
    out = fresh(root/('controls_viewer' if controls else 'viewer'))
    records = {r['key']: r for r in result['records']}
    selections = {(s['case_key'], s['policy']): s for s in result['selections']}
    cases, audio = [], {}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if 'J' in arrays and 'parents' in arrays:
            J, parents = arrays['J'], arrays['parents']
        else:
            with np.load(root/'population'/'arrays.npz', allow_pickle=False) as skeleton:
                J, parents = skeleton['J'].copy(), skeleton['parents'].copy()
        for case in result['cases']:
            panels = [_panel(p, selections.get((case['key'], p)), records, arrays, J, parents)
                      for p in policies]
            cases.append(dict(key=case['key'], role=case['role'],
                source_pair={s: _source(case.get(s)) for s in ('left', 'right')},
                music_interval=case['music_interval'], music_name='dunhuangwu%d' % (int(case.get('songidx', case.get('song', 0)))+2),
                beats_local=[float(b)-case['music_interval'][0] for b in case['beats']
                             if case['music_interval'][0] <= b < case['music_interval'][1]],
                audio=_audio(case, out, audio), panels=panels,
                exposure=case.get('evaluation_exposure', 'HISTORICALLY_EXPOSED')))
    payload = dict(schema='R63_ACTUAL_FIXED_BUDGET_VIEWER_V1', fps=FPS, frames=FRAMES,
        duration_seconds=FRAMES/FPS, cases=cases, case_count=len(cases), policies=list(policies),
        parents=np.asarray(parents).astype(int).tolist(), controls=bool(controls),
        main_result_sha256=sha(root/'main'/'result.json'), displayed_result_sha256=sha(result_path),
        displayed_arrays_sha256=sha(arrays_path), binding=result.get('binding'),
        clock_contract='ONE_AUDIO_MASTER_ACTUAL528_BODY22_ALL_ARMS_SAME_MUSIC_TIME',
        world_root_contact='UNKNOWN', Native24='NOT_EVALUABLE', human_reviewed=False,
        view='ROOT_RELATIVE_BODY22_ONLY_NO_WORLD_TRAJECTORY',
        displayed_coordinates='ACTUAL_SAVED_ROTATIONS_FK_FLOAT32_NO_FILTER_NO_RETIMING',
        failed_selection='BLANK_NO_FALLBACK')
    write(out/'data.json', payload)
    protocol = dict(schema=payload['schema'], fps=FPS, frames=FRAMES, duration_seconds=FRAMES/FPS,
        policies=list(policies), all_registered_cases_included=True, case_count=len(cases),
        source_roles={role: sum(c['role'] == role for c in cases) for role in ('DEV', 'EVAL')},
        coordinate_scope=payload['view'], world_camera_available=False,
        clock_contract=payload['clock_contract'], failed_selection=payload['failed_selection'],
        audio_clips=list(audio.values()), main_result_sha256=payload['main_result_sha256'],
        displayed_result_sha256=payload['displayed_result_sha256'],
        actual_body22_data=True, original_source_rotations_in_packet=False,
        includes_audio_excerpt=True, contains_models=False, contains_private_run_records=False,
        human_reviewed=False, Native24='NOT_EVALUABLE',
        acceptance_dom=['#case', '#play', '#prev', '#next', '#seek', '#rate', '#clock',
                        '#masterAudio', '#panels', '#details', '#evidence'],
        frame_contract='frame=floor(audio.currentTime*30), clamped0..527; step seeks frame/30',
        silent_clock='MONOTONIC_PERFORMANCE_CLOCK_ONLY_WHEN_EXPLICITLY_MUTED',
        camera_contract='ONE_FIXED_SCALE_AND_VIEW_FOR_ALL_ARMS_OF_SELECTED_CASE')
    write(out/'protocol.json', protocol)
    template = (HERE/'viewer.html').read_text(encoding='utf8')
    # Embedded data lets the same approved packet open directly on file://.
    # data.json is also kept as a separately hashed machine-readable artifact.
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    (out/'index.html').write_text(template.replace('__R63_PAYLOAD__', encoded.replace('</', '<\\/')),
                                  encoding='utf8')
    files = [dict(path=str(p.relative_to(out)).replace('\\', '/'), bytes=p.stat().st_size,
                  sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    manifest = dict(schema='R63_WATCH_PACKET_V1', root=str(out), files=files,
                    total_files=len(files), total_bytes=sum(p['bytes'] for p in files),
                    includes_body22_coordinates=True, includes_audio_excerpt=True,
                    includes_rotation_npz=False, includes_weights=False, includes_dataset=False,
                    result_sha256=sha(result_path), controls=bool(controls))
    write(out/'packet_manifest.json', manifest)
    return manifest
