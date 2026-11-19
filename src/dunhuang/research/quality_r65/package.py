"""Seal a genuinely anonymous reviewer packet from the unchanged R64 outputs."""
import copy
import hashlib
import hmac
import random
import shutil
import wave
from collections import Counter
from pathlib import Path

import numpy as np

from .common import CFG, HERE, PARENT, canonical, digest, fresh, read, sha, write
from dunhuang.research.budget_r63.viewer import _panel


def opaque(seed, namespace, value):
    return hmac.new(bytes.fromhex(seed), (namespace+':'+value).encode(), hashlib.sha256).hexdigest()[:24]


def assignments(seed, cases, music):
    """Private random order, balanced positions; no canonical method list in packet."""
    rng = random.Random(int(opaque(seed, 'rng', 'forms'), 16))
    forms = []
    for index in range(CFG['review_forms']):
        order = [c['blind_item_id'] for c in cases]
        rng.shuffle(order)
        methods = list(CFG['methods']); rng.shuffle(methods)
        panel_orders = {}
        lookup = {c['blind_item_id']: c for c in cases}
        for i, case_id in enumerate(order):
            by_method = {v['method']: v['blind_variant_id'] for v in lookup[case_id]['variants']}
            rotation = methods[i % 4:]+methods[:i % 4]
            panel_orders[case_id] = [by_method[m] for m in rotation]
        songs = [m['blind_item_id'] for m in music]; rng.shuffle(songs)
        forms.append(dict(form_id='F%02d' % (index+1), case_order=order,
                          panel_orders=panel_orders, music_order=songs))
    return forms


def leak_check(payload):
    """Reject private fields even if a future template would visually hide them."""
    forbidden = {'method', 'policy', 'role', 'case_key', 'source_uid', 'event_uid',
                 'record_key', 'metrics', 'proxies', 'regions', 'seed', 'source_interval',
                 'music_interval', 'time_warp', 'allocation', 'trajectory_sha256'}
    def walk(value):
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise ValueError('Private reviewer key: '+str(forbidden.intersection(value)))
            for v in value.values(): walk(v)
        elif isinstance(value, list):
            for v in value: walk(v)
        elif isinstance(value, str) and value in CFG['methods']:
            raise ValueError('Method identity in reviewer packet')
    walk(payload)
    return True


def public_order(cases):
    """Scrub construction order as well as names; canonical order is a label leak."""
    rows=copy.deepcopy(cases)
    for row in rows:
        row['variants'].sort(key=lambda v:v['blind_variant_id'])
    return sorted(rows,key=lambda c:c['blind_item_id'])


def seal_inputs(parent=PARENT):
    parent = Path(parent)
    manifest = read('reports/r64_release/public_manifest.json')
    if sha(parent/'main/result.json') != CFG['parent_main_sha256'] or sha(parent/'controls/result.json') != CFG['parent_controls_sha256']:
        raise ValueError('R64 frozen results changed')
    if sha('reports/r64_release/scientific_files.json') != manifest['scientific_files_sha256']:
        raise ValueError('R64 public scientific seal changed')
    rows = []
    for r in read('reports/r64_release/scientific_files.json')['files']:
        p = (parent/r['path']).resolve()
        if parent.resolve() not in p.parents or sha(p) != r['sha256']:
            raise ValueError('Frozen R64 artifact mismatch: '+str(p))
        rows.append(dict(path=str(p), sha256=r['sha256']))
    for name in ('timing_r64', 'budget_r63'):
        for p in sorted((HERE.parent/name).iterdir()):
            if p.suffix in ('.py', '.json', '.html'):
                rows.append(dict(path=str(p.resolve()), sha256=sha(p)))
    return rows


def check_seal(rows):
    for r in rows:
        if sha(r['path']) != r['sha256']:
            raise ValueError('Frozen parent modified: '+r['path'])
    return len(rows)


def build(out, seed, parent=PARENT):
    """No new solves and no selection: all registered methods/cases kept intact."""
    if len(seed) != 64:
        raise ValueError('Private 256-bit assignment seed required')
    bytes.fromhex(seed)
    out = fresh(out); review = fresh(out/'reviewer'); private = fresh(out/'admin')
    parent = Path(parent)
    source_path = parent/'controls/result.json'
    result = read(source_path)
    if sha(source_path) != CFG['parent_controls_sha256']:
        raise ValueError('Wrong R64 control result')
    protocol_sha = sha(HERE/'protocol.json')
    pack_id = digest(dict(protocol_sha256=protocol_sha, source_sha256=sha(source_path),
                          assignment_seed_sha256=hashlib.sha256(bytes.fromhex(seed)).hexdigest()))
    admin_cases, public_cases, songs, media = [], [], {}, {}
    frozen_clips=read(parent/'viewer/protocol.json')['audio_clips']
    records = {r['key']: r for r in result['records']}
    selected = {(s['case_key'], s['policy']): s for s in result['selections']}
    arrays_seal, parents_expected = {}, None
    for case in result['cases']:
        key = case['key']; case_id = opaque(seed, 'case', key)
        song = int(case['songidx']); music_id = opaque(seed, 'music', str(song))
        path = (parent/result['array_files'][key]).resolve()
        if parent.resolve() not in path.parents:
            raise ValueError('Array path escapes registered root')
        arrays_seal[key] = sha(path)
        variants, visible = [], []
        with np.load(path, allow_pickle=False) as arrays:
            J, parents = arrays['J'], arrays['parents']
            if parents_expected is None: parents_expected = parents.copy()
            if not np.array_equal(parents_expected, parents): raise ValueError('Skeleton changed')
            for method in CFG['methods']:
                selection = selected[(key, method)]
                panel = _panel(method, selection, records, arrays, J, parents)
                rec = records.get(selection['record_key'])
                variant_id = opaque(seed, 'variant', key+':'+method)
                m = rec['music_reward'] if rec else {}
                variants.append(dict(blind_variant_id=variant_id, method=method,
                    record_key=selection['record_key'], trajectory_sha256=panel.get('rotation_sha256'),
                    coordinate_sha256=panel.get('coordinates_sha256'), available=bool(panel['data']),
                    proxies={a: m.get(b) for a,b in [('beat','phase_beat_score'),('accent','phase_accent_score'),
                        ('section','section_proxy_score'),('primary','primary_music_score')]},
                    original_screen_passed=panel.get('numerical_screen',False)))
                visible.append(dict(blind_variant_id=variant_id, data=panel['data'],
                                    bounds=panel.get('bounds'), available=bool(panel['data'])))
        A, B = case['music_interval']
        matches=[a for a in frozen_clips if a['source_audio_sha256']==case['audio_sha256']
                 and a['source_frame_interval']==[A,B]]
        if len(matches)!=1:raise ValueError('Exact frozen R64 viewing audio required')
        clip=matches[0]
        clip_source=(parent/'viewer'/clip['path']).resolve()
        if (parent/'viewer').resolve() not in clip_source.parents or sha(clip_source)!=clip['sha256']:
            raise ValueError('Frozen music excerpt changed')
        clip_name='audio/'+opaque(seed,'clip',clip['path'])+'.m4a'
        clip_dest=review/clip_name;clip_dest.parent.mkdir(exist_ok=True)
        if not clip_dest.exists():shutil.copy2(str(clip_source),str(clip_dest))
        clip_public = dict(path=clip_name, media_sha256=clip['sha256'])
        admin_cases.append(dict(blind_item_id=case_id, case_key=key, role=case['role'],
            music_id=music_id, music_interval_seconds=[A/30,B/30],
            source_uids=[case[s]['source_uid'] for s in ('left','right')], variants=variants,
            clip=dict(clip, reviewer_path=clip_name,source_origin_seconds=A/30, source_origin_frame=A)))
        public_cases.append(dict(blind_item_id=case_id, audio=clip_public,
                                 variants=sorted(visible,key=lambda v:v['blind_variant_id'])))
        if song not in songs:
            raw = Path(case['audio_path'])
            if sha(raw) != case['audio_sha256']: raise ValueError('Actual full audio changed')
            name = 'audio/'+music_id+'.wav'; dest = review/name; dest.parent.mkdir(exist_ok=True)
            shutil.copy2(str(raw),str(dest))
            with wave.open(str(dest),'rb') as wav:
                samples, sr = wav.getnframes(), wav.getframerate()
                channels, width = wav.getnchannels(), wav.getsampwidth()
            clock = result['music_clocks'][str(song)]
            if samples != clock['audio_samples'] or sr != clock['sample_rate']:
                raise ValueError('Raw sample clock differs from R64')
            songs[song] = dict(blind_item_id=music_id, song_index=song,
                raw_audio_sha256=sha(raw), media_sha256=sha(dest), media_path=name,
                duration_seconds=samples/sr, samples=samples, sample_rate=sr,
                channels=channels,sample_width_bytes=width,source_sample_interval=[0,samples],
                media_origin_sample=0,source_origin_sample=0,transcoding=False,
                audio_path=str(raw),clock_sha256=clock['clock_sha256'])
    if len(admin_cases) != 29 or len(songs) != 3:
        raise ValueError('All 29 cases and three full music inputs must be retained')
    music = [songs[s] for s in sorted(songs)]
    forms = assignments(seed, admin_cases, music)
    admin = dict(schema='R65_PRIVATE_ADMIN_V1', pack_id=pack_id, protocol_sha256=protocol_sha,
        parent_controls_sha256=sha(source_path), forms=forms, cases=admin_cases, music=music,
        input_array_sha256_by_case=arrays_seal, private_mapping=True,
        all_historically_exposed=True, semantic_event_truth='UNKNOWN', Native24='NOT_EVALUABLE')
    payload = dict(pack_id=pack_id,protocol_sha256=protocol_sha,fps=30,frames=528,
        parents=parents_expected.astype(int).tolist(), dimensions=CFG['dimensions'], forms=forms,
        cases=public_order(public_cases), music=sorted([dict(blind_item_id=m['blind_item_id'],
            audio=dict(path=m['media_path'],media_sha256=m['media_sha256']),
            duration_seconds=m['duration_seconds']) for m in music],key=lambda m:m['blind_item_id']))
    leak_check(payload)
    from .viewer import write_viewers
    write_viewers(review,payload)
    write(review/'protocol.json',dict(schema='R65_REVIEWER_PREPARATION_V1',pack_id=pack_id,
        protocol_sha256=protocol_sha,case_count=29,panels_per_case=4,full_audio_items=3,
        human_records=0,human_reviewed=False,methods_blinded=True,administrator_map_in_packet=False,
        actual_body22_coordinates=True,full_music_annotation_audio_untranscoded=True,
        motion_music_excerpts_transcoded=True,
        no_automatic_music_marks=True,Native24='NOT_EVALUABLE',can_route=False))
    # No raw reviewer payload file: coordinates embedded in one watch page only.
    write(private/'mapping.json',admin)
    counts = {}
    for form in forms:
        lookup = {c['blind_item_id']:c for c in admin_cases}; count = Counter()
        for cid, order in form['panel_orders'].items():
            identities = {v['blind_variant_id']:v['method'] for v in lookup[cid]['variants']}
            for position, vid in enumerate(order): count[(identities[vid],position)] += 1
        counts[form['form_id']] = {method:[count[(method,i)] for i in range(4)] for method in CFG['methods']}
    files = [dict(path=p.relative_to(review).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
             for p in sorted(review.rglob('*')) if p.is_file()]
    manifest = dict(schema='R65_REVIEWER_PACKET_V1',pack_id=pack_id,files=files,
        total_files=len(files),total_bytes=sum(r['bytes'] for r in files),
        contains_body22_coordinates=True,contains_rotation_arrays=False,contains_models=False,
        contains_full_original_audio=True,administrator_mapping_in_packet=False)
    write(out/'reviewer_manifest.json',manifest)
    duplicates = []
    for case in admin_cases:
        by_hash = {}
        for v in case['variants']:
            if v['trajectory_sha256']:
                by_hash.setdefault(v['trajectory_sha256'],[]).append(v['method'])
        duplicates.extend(dict(blind_item_id=case['blind_item_id'],methods=m,trajectory_sha256=h)
                          for h,m in by_hash.items() if len(m)>1)
    summary = dict(pack_id=pack_id,case_count=29,method_outputs=116,music_count=3,
        assignment_position_counts=counts,identical_trajectory_groups=duplicates,
        zero_new_geometry_solves=True,zero_neural_training_steps=True,
        status='PREPARATION_ONLY_NO_HUMAN_LABELS',Native24='NOT_EVALUABLE',can_route=False)
    write(out/'preparation.json',summary)
    return admin, manifest, summary
