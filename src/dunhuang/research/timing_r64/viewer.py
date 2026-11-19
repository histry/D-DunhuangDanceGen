"""R64 watch packet; reuse frozen playback, extend only timing diagnostics.

The source NPZs stay private. The public packet is HTML+protocol+audio excerpts;
coordinates are embedded once, and all failed selections remain blank.
"""
import json
from pathlib import Path

import numpy as np

from .common import ROOT, read, write, sha, fresh
from dunhuang.research.budget_r63 import viewer as frozen


MAIN_POLICIES = ('Fixed48', 'LocalBeat')
CONTROL_POLICIES = ('Fixed48', 'LocalShortest', 'LocalBalanced', 'LocalBeat',
                    'ShuffledBeat', 'Fixed24')
FPS, FRAMES = 30, 528


def music_clock(clock, case):
    """Slice actual acoustic proxies to the common target clock; no truth upgrade."""
    A, B = map(int, case['music_interval'])
    def events(key):
        values = clock.get(key)
        return [float(t)-A for t in (values if values is not None else []) if A <= float(t) < B]
    def dense(key):
        values = clock.get(key)
        if values is None:
            return []
        if len(values) < B:
            raise ValueError('Full-track '+key+' does not cover the actual music interval')
        x = np.asarray(values[A:B], float)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError('Finite30Hz acoustic proxy curve required')
        return x.tolist()
    beats = events('beat_frames') if clock.get('beat_frames') is not None else [
        float(t)-A for t in case.get('beats', []) if A <= float(t) < B]
    return dict(beat_frames=beats, accent_strength=dense('accent_strength'),
        onset_strength=dense('onset_strength'), section_proxy_frames=events('section_proxy_frames'),
        phrase_proxy_frames=events('phrase_proxy_frames'), source_frame_interval=[A, B], fps=FPS,
        source_audio_sha256=clock.get('audio_sha256', case.get('audio_sha256')),
        downbeat_frames=None, independent_phrase_labels=None,
        downbeat_status='UNKNOWN', independent_phrase_status='UNKNOWN',
        section_scope='ACOUSTIC_BOUNDARY_PROXY_NOT_INDEPENDENT_PHRASE_TRUTH',
        accent_scope='ACOUSTIC_ACCENT_PROXY_NOT_HUMAN_MUSIC_ANNOTATION',
        algorithm=clock.get('algorithm', clock.get('schema', 'REGISTERED_ACOUSTIC_CLOCK')))


EXTRA_HTML = '''<div class="toolbar" style="margin-top:10px"><label for="sourceClock">帧读数</label><select id="sourceClock"><option value="target">共同目标音乐帧</option><option value="source">来源帧映射（当前重定时预览）</option></select><span class="muted">重拍：UNKNOWN · 独立乐句：UNKNOWN；以下仅为声学代理，非人工结构真值。</span></div><canvas id="musicTimeline" style="width:100%;height:116px;background:white;border-radius:9px;margin-top:10px" aria-label="实际音乐代理时间线"></canvas>'''


EXTRA_JS = r'''
function sourceFrameReadouts(){if($('sourceClock').value!=='source')return;current.panels.forEach((p,i)=>{const out=$('frame'+i);if(!p.data){out.textContent='无合格输出 · 未替换';return}const region=p.regions.find(r=>r.interval[0]<=frame&&frame<r.interval[1]);if(!region||region.kind==='TRANSITION'){out.textContent='目标帧 '+frame+' · Geometry生成区 · 无真实来源对应';return}const side=region.kind==='LEFT_CORE'?'left':'right',meta=p.source_clock[side],q=meta.source_positions,source=q[frame-region.interval[0]];out.textContent='目标帧 '+frame+' → '+meta.source_uid+' 来源帧 '+Number(source).toFixed(3)+' ('+(source/30).toFixed(3)+' s)'});}
function drawMusicTimeline(){const cv=$('musicTimeline'),[ctx,w,h]=setup(cv),clock=current.music_clock;ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);const margin=145,right=10,span=w-margin-right,x=t=>margin+t/D.frames*span;ctx.font='12px system-ui';ctx.textAlign='left';ctx.fillStyle='#506c74';ctx.fillText('检测拍点',8,21);ctx.fillText('声学重音代理',8,55);ctx.fillText('声学边界代理',8,88);ctx.fillStyle='#2b7b67';for(const t of clock.beat_frames){ctx.fillRect(x(t),8,1,20)}const accent=clock.accent_strength.length?clock.accent_strength:clock.onset_strength,max=Math.max(...accent,1e-9);ctx.strokeStyle='#b9863f';ctx.beginPath();accent.forEach((v,i)=>{const y=64-v/max*28;i?ctx.lineTo(x(i),y):ctx.moveTo(x(i),y)});ctx.stroke();ctx.fillStyle='#775c9c';for(const t of clock.section_proxy_frames){ctx.beginPath();ctx.arc(x(t),83,4,0,Math.PI*2);ctx.fill()}ctx.strokeStyle='#775c9c';for(const t of clock.phrase_proxy_frames){ctx.beginPath();ctx.moveTo(x(t),91);ctx.lineTo(x(t),99);ctx.stroke()}ctx.strokeStyle='#aa3832';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x(frame),3);ctx.lineTo(x(frame),104);ctx.stroke();ctx.fillStyle='#506c74';ctx.fillText('0 s',margin,113);ctx.fillText('17.6 s',w-52,113);window.R64_EVIDENCE.music_proxy_scope=clock.section_scope;window.R64_EVIDENCE.downbeat_status='UNKNOWN';window.R64_EVIDENCE.independent_phrase_status='UNKNOWN';window.R64_EVIDENCE.source_clock_mode=$('sourceClock').value;}
'''


def template():
    """Leave frozen R63 source untouched, retaining its actual audio/frame loop."""
    text = (frozen.HERE/'viewer.html').read_text(encoding='utf8')
    text = text.replace('R63', 'R64').replace('r63-data', 'r64-data')
    text = text.replace('如何分配动作与过渡？', '局部非对称时间候选')
    text = text.replace('<main id="panels">', EXTRA_HTML+'<main id="panels">')
    anchor = "for(const[c,i]of D.cases.map((c,i)=>[c,i]))"
    if anchor not in text:
        raise ValueError('Frozen playback template changed; inspect before adapting')
    text = text.replace(anchor, EXTRA_JS+'\n'+anchor)
    text = text.replace('current.panels.forEach((_,i)=>drawPanel(i));',
        'current.panels.forEach((_,i)=>drawPanel(i));sourceFrameReadouts();drawMusicTimeline();')
    text = text.replace("$('case').addEventListener('change',load);",
        "$('sourceClock').addEventListener('change',()=>{lastDraw=-1;draw()});"
        "$('musicTimeline').addEventListener('click',event=>{const box=event.currentTarget.getBoundingClientRect();changeFrame((event.clientX-box.left-145)/(box.width-155)*D.frames)});"
        "$('case').addEventListener('change',load);")
    text = text.replace('真实 FK，无显示滤波', '真实 FK，无显示滤波')
    return text


def _array_path(root, relative):
    path = (root/relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ValueError('Registered case arrays must remain inside R64 output root')
    return path


def viewer(root=ROOT, controls=False):
    """Write a self-contained actual-trajectory packet and return safe manifest.

    The manifest is root/viewer_manifest.json (or controls_viewer_manifest.json),
    outside the watch directory. No duplicate data.json is emitted.
    """
    root = Path(root)
    result_path = root/('controls' if controls else 'main')/'result.json'
    result = read(result_path)
    out = fresh(root/('controls_viewer' if controls else 'viewer'))
    policies = CONTROL_POLICIES if controls else MAIN_POLICIES
    records = {r['key']: r for r in result['records']}
    selections = {(s['case_key'], s['policy']): s for s in result['selections']}
    cases, audio, array_hashes = [], {}, {}
    parents_expected = None
    clocks = result.get('music_clocks', {})
    for case in result['cases']:
        path = _array_path(root, result['array_files'][case['key']])
        array_hashes[case['key']] = sha(path)
        with np.load(path, allow_pickle=False) as arrays:
            J, parents = arrays['J'], arrays['parents']
            if parents_expected is None:
                parents_expected = parents.copy()
            elif not np.array_equal(parents_expected, parents):
                raise ValueError('Playback skeleton parents changed between case files')
            panels = [frozen._panel(p, selections.get((case['key'], p)), records,
                                    arrays, J, parents) for p in policies]
        for panel in panels:
            rec = records.get(panel.get('record_key'))
            if rec is not None:
                panel.update(timing_offset_frames=rec.get('timing_offset_frames'),
                    numerical_proxy_diagnostics={k: rec[k] for k in (
                        'scores', 'music_scores', 'musical', 'music_proxy',
                        'duration_cost', 'boundary_cost', 'balance_cost', 'music_cost',
                        'retention_guard', 'guard_qualified', 'balanced_cost', 'music_reward',
                        'music_reward_shuffled', 'retention_guard_passed', 'section_relation_proxy') if k in rec},
                    score_scope='REGISTERED_PROXY_OR_COST_NOT_QUALITY_PROBABILITY',
                    acceptance_scope='FROZEN_G1_NUMERICAL_SCREEN_ONLY_NOT_NATIVE24')
        song = int(case.get('songidx', case.get('song', 0)))
        clock = clocks.get(str(song), clocks.get(song, {}))
        local = music_clock(clock, case)
        cases.append(dict(key=case['key'], role=case['role'],
            source_pair={s: frozen._source(case.get(s)) for s in ('left', 'right')},
            music_interval=case['music_interval'], music_name='dunhuangwu%d' % (song+2),
            beats_local=local['beat_frames'], music_clock=local,
            audio=frozen._audio(case, out, audio), panels=panels,
            exposure=case.get('evaluation_exposure', 'HISTORICALLY_EXPOSED')))
    if parents_expected is None:
        raise ValueError('Viewer population must retain all registered cases')
    payload = dict(schema='R64_ACTUAL_LOCAL_TIMING_VIEWER_V1', fps=FPS, frames=FRAMES,
        duration_seconds=FRAMES/FPS, cases=cases, case_count=len(cases), policies=list(policies),
        parents=parents_expected.astype(int).tolist(), controls=bool(controls),
        main_result_sha256=sha(root/'main'/'result.json'), displayed_result_sha256=sha(result_path),
        displayed_arrays_sha256_by_case=array_hashes, binding=result.get('binding'),
        clock_contract='ONE_AUDIO_MASTER_ACTUAL528_BODY22_ALL_ARMS_SAME_MUSIC_TIME',
        world_root_contact='UNKNOWN', Native24='NOT_EVALUABLE', human_reviewed=False,
        view='ROOT_RELATIVE_BODY22_ONLY_NO_WORLD_TRAJECTORY',
        displayed_coordinates='ACTUAL_SAVED_ROTATIONS_FK_FLOAT32_NO_FILTER_NO_RETIMING',
        failed_selection='BLANK_NO_FALLBACK', downbeat_status='UNKNOWN',
        independent_phrase_status='UNKNOWN', BalancedEqual='STATISTICS_ONLY_NOT_DISPLAYED')
    protocol = dict(schema=payload['schema'], fps=FPS, frames=FRAMES, duration_seconds=FRAMES/FPS,
        policies=list(policies), all_registered_cases_included=True, case_count=len(cases),
        source_roles={r: sum(c['role'] == r for c in cases) for r in ('DEV', 'EVAL')},
        coordinate_scope=payload['view'], world_camera_available=False,
        clock_contract=payload['clock_contract'], failed_selection=payload['failed_selection'],
        audio_clips=list(audio.values()), main_result_sha256=payload['main_result_sha256'],
        displayed_result_sha256=payload['displayed_result_sha256'],
        actual_body22_data=True, original_source_rotations_in_packet=False,
        contains_duplicate_data_json=False, includes_audio_excerpt=True,
        contains_models=False, contains_private_run_records=False,
        downbeat_status='UNKNOWN', independent_phrase_status='UNKNOWN',
        structure_scope='ACOUSTIC_SECTION_AND_PHRASE_PROXIES_ONLY_NOT_ANNOTATED_HIERARCHY',
        quality_probability_available=False, acceptance_scope='FROZEN_G1_NUMERICAL_SCREEN_ONLY',
        human_reviewed=False, Native24='NOT_EVALUABLE',
        acceptance_dom=['#case', '#play', '#prev', '#next', '#seek', '#rate', '#clock',
                        '#masterAudio', '#panels', '#details', '#evidence', '#musicTimeline', '#sourceClock'],
        frame_contract='frame=floor(audio.currentTime*30), clamped0..527; step seeks frame/30',
        source_clock_selector='READOUT_ONLY_RETIMED_PREVIEW_NOT_ORIGINAL_SPEED_PLAYBACK',
        silent_clock='MONOTONIC_PERFORMANCE_CLOCK_ONLY_WHEN_EXPLICITLY_MUTED',
        camera_contract='ONE_FIXED_SCALE_AND_VIEW_FOR_ALL_ARMS_OF_SELECTED_CASE')
    write(out/'protocol.json', protocol)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    (out/'index.html').write_text(template().replace('__R64_PAYLOAD__',
        encoded.replace('</', '<\\/')), encoding='utf8')
    files = [dict(path=str(p.relative_to(out)).replace('\\', '/'), bytes=p.stat().st_size,
                  sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    manifest = dict(schema='R64_WATCH_PACKET_V1', root=str(out), files=files,
        total_files=len(files), total_bytes=sum(p['bytes'] for p in files),
        includes_body22_coordinates=True, includes_audio_excerpt=True,
        includes_rotation_npz=False, includes_weights=False, includes_dataset=False,
        result_sha256=sha(result_path), controls=bool(controls))
    write(root/('controls_viewer_manifest.json' if controls else 'viewer_manifest.json'), manifest)
    return manifest
