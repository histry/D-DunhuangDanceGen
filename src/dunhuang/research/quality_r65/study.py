"""Effect-first preparation, frozen-score controls, and independent exact replay."""
import copy
import secrets
import shutil
from datetime import datetime
from pathlib import Path

from .common import CFG, HERE, ROOT, PARENT, binding, canonical, digest, fresh, read, sha, write
from .package import build, seal_inputs, check_seal


def documents_from(path):
    if path is None:
        return []
    path = Path(path)
    if not path.is_dir():
        raise ValueError('Explicit actual-review directory required')
    documents = []
    for p in sorted(path.glob('*.json')):
        doc = read(p)
        if str(doc.get('reviewer_id','')).startswith('UI_ACCEPTANCE_TEST') or doc.get('test_fixture'):
            raise ValueError('UI/test fixtures must never enter actual review analysis')
        documents.append(doc)
    return documents


def roster_from(path):
    if path is None:
        return None
    roster = read(path)
    for r in roster.get('entries',[]):
        evidence = r.get('authentication_evidence_path')
        if not evidence or not Path(evidence).is_file() or sha(evidence) != r.get('authentication_evidence_sha256'):
            raise ValueError('Roster evidence missing or changed; self-declared IDs are not authentication')
        if r.get('audio_before_motion_verified'):
            order=r.get('order_evidence_path')
            if not order or not Path(order).is_file() or sha(order)!=r.get('order_evidence_sha256'):
                raise ValueError('Music sealing/access evidence missing or changed')
    return roster


def quality_order(quality, music_documents, admin, roster):
    """Client export timestamps alone cannot attest an audio-first review."""
    from .records import qualify
    entries=[] if roster is None else [r for r in roster.get('entries',[]) if r.get('reviewer_id')==quality['reviewer_id']]
    if len(entries)!=1:return False,'NO_VERIFIED_AUDIO_FIRST_ADMIN_RECORD'
    entry=entries[0]
    if not entry.get('audio_before_motion_verified'):return False,'AUDIO_FIRST_NOT_ADMIN_VERIFIED'
    candidates=[d for d in music_documents if d['review_id']==entry.get('music_review_id')
                and d['reviewer_id']==quality['reviewer_id'] and d['form_id']==quality['form_id']]
    if len(candidates)!=1:return False,'SEALED_AUDIO_DOCUMENT_MISSING_OR_AMBIGUOUS'
    doc=candidates[0]
    if digest(doc)!=entry.get('music_review_digest'):return False,'SEALED_AUDIO_DOCUMENT_CHANGED'
    if not qualify(doc,admin,roster)['contract_qualified']:return False,'AUDIO_REVIEW_INDEPENDENCE_UNVERIFIED'
    music={m['blind_item_id']:m for m in admin['music']}
    if set(i['blind_item_id'] for i in doc['items'])!=set(music):return False,'AUDIO_REVIEW_POPULATION_INCOMPLETE'
    for item in doc['items']:
        if item['coverage_seconds']!=[0,music[item['blind_item_id']]['duration_seconds']]:return False,'AUDIO_REVIEW_COVERAGE_INCOMPLETE'
        if any(t['status']=='PARTIAL' for t in item['tracks'].values()):return False,'AUDIO_REVIEW_TASK_UNFINISHED'
    try:
        sealed=datetime.fromisoformat(entry['music_sealed_at'].replace('Z','+00:00'))
        opened=datetime.fromisoformat(entry['motion_first_access_at'].replace('Z','+00:00'))
        if sealed.utcoffset() is None or opened.utcoffset() is None or sealed>=opened:raise ValueError()
    except (KeyError,TypeError,ValueError):return False,'ADMIN_AUDIO_SEAL_ACCESS_CLOCK_INVALID'
    return True,None


def review_analysis(admin, documents, roster=None):
    from .records import validate_document
    from .metrics import summarize_reviews, event_match
    docs = [validate_document(d,admin) for d in documents]
    # An administrator must seal the music record before first motion access;
    # a client-supplied reviewed_at timestamp is insufficient proof of ordering.
    audio_documents=[d for d in docs if d['schema']=='R65_MUSIC_REVIEW_V1']
    evaluation, order_exclusions = [], []
    for d in docs:
        if d['schema'] == 'R65_QUALITY_REVIEW_V1':
            allowed,reason=quality_order(d,audio_documents,admin,roster)
            if not allowed:
                order_exclusions.append(dict(review_id=d['review_id'],reason=reason))
                continue
        evaluation.append(d)
    stats = summarize_reviews(admin,evaluation,roster,tolerances=CFG['match_tolerance_seconds'])
    # Automatic-vs-human comparison is per qualified person, type and actual interval.
    from .records import qualify
    clocks = read(PARENT/'main/music_clocks.json') if documents else {}
    music = {m['blind_item_id']:m for m in admin['music']}
    comparisons = []
    for doc in evaluation:
        if doc['schema'] != 'R65_MUSIC_REVIEW_V1' or not qualify(doc,admin,roster)['contract_qualified']:
            continue
        for item in doc['items']:
            m = music[item['blind_item_id']];clock = clocks[str(m['song_index'])]
            for kind, track in item['tracks'].items():
                if track['status'] not in ('COMPLETE','NO_CLEAR_EVENTS'):continue
                candidates = {'beat':[('automatic_beat_candidates','beat_frames')],
                    'accent':[('acoustic_accent_proxy','accent_frames')],
                    'phrase':[('section_proxy_not_phrase_truth','section_proxy_frames'),
                              ('chroma_proxy_not_phrase_truth','phrase_proxy_frames')],
                    'downbeat':[]}[kind]
                if not candidates:
                    comparisons.append(dict(reviewer_id=doc['reviewer_id'],music_id=m['blind_item_id'],
                        kind=kind,status='NOT_EVALUABLE_NO_AUTOMATIC_DOWNBEAT',metrics=None))
                for label, field in candidates:
                    result = event_match([e['seconds'] for e in track['events']],
                        [float(f)/30 for f in clock[field]],CFG['match_tolerance_seconds'][kind],
                        item['coverage_seconds'],excluded=[(q['start'],q['end']) for q in item['unsure_intervals']])
                    comparisons.append(dict(reviewer_id=doc['reviewer_id'],music_id=m['blind_item_id'],
                        kind=kind,candidate_scope=label,metrics=result,
                        comparison_scope='AGAINST_ONE_INDEPENDENT_REVIEWER_NOT_ADJUDICATED_TRUTH'))
    return dict(schema='R65_ACTUAL_REVIEW_ANALYSIS_V1',stats=stats,
        automatic_vs_human=comparisons,actual_input_documents=len(documents),
        actual_review_identities=len({d['reviewer_id'] for d in docs}),
        original_documents_preserved=True,independence_order_checked=True,order_exclusions=order_exclusions,
        interpretation='NO_RECORDS_MEANS_NOT_EVALUABLE; CONTRACT_CHECKS_ARE_NOT_CULTURAL_OR_PHYSICAL_TRUTH',
        Native24='NOT_EVALUABLE',can_route=False)


def prepare():
    seal = seal_inputs()
    root = fresh(ROOT)
    seed = secrets.token_hex(32)
    write(root/'private_seed.json',dict(seed=seed,private=True))
    admin, manifest, preparation = build(root/'packet',seed)
    for item in admin['music']:
        seal.append(dict(path=str(Path(item['audio_path']).resolve()),sha256=item['raw_audio_sha256']))
    protocol=PARENT/'viewer/protocol.json';seal.append(dict(path=str(protocol.resolve()),sha256=sha(protocol)))
    for clip in read(protocol)['audio_clips']:
        seal.append(dict(path=str((PARENT/'viewer'/clip['path']).resolve()),sha256=clip['sha256']))
    empty = review_analysis(admin,[])
    write(root/'evaluation_empty.json',empty)
    result = dict(binding=binding(),protocol_sha256=sha(HERE/'protocol.json'),
        parent_main_sha256=CFG['parent_main_sha256'],parent_controls_sha256=CFG['parent_controls_sha256'],
        preparation=preparation,reviewer_manifest_sha256=sha(root/'packet/reviewer_manifest.json'),
        admin_mapping_sha256=sha(root/'packet/admin/mapping.json'),
        empty_evaluation_sha256=sha(root/'evaluation_empty.json'),input_seal=seal,
        frozen_parent_files_unchanged=check_seal(seal),human_reviewed=False,
        human_quality_metrics='NOT_EVALUABLE_NO_VERIFIED_INDEPENDENT_RECORDS',
        new_geometry_solves=0,neural_training_steps=0,optimizer_calls=0,full_song_runs=0,
        can_route=False,Native24='NOT_EVALUABLE')
    write(root/'main.json',result)
    print('R65_PREPARATION_READY_29_CASES_116_FROZEN_OUTPUTS_ZERO_HUMAN_LABELS',flush=True)
    return result


def controls(gate=True):
    if gate:
        receipt = read(ROOT/'receipts/main_browser.json')
        if not receipt.get('passed') or receipt['main_result_sha256'] != sha(ROOT/'main.json'):
            raise ValueError('Actual main viewer delivery and browser receipt required before controls')
    out = fresh(ROOT/'controls')
    admin = read(ROOT/'packet/admin/mapping.json')
    source = read(PARENT/'controls/result.json')
    keys = [v['record_key'] for c in admin['cases'] for v in c['variants']]
    from .diagnostics import batch
    result = batch(source['records'],source['cases'],source['music_clocks'],keys)
    result.update(binding=binding(),parent_controls_sha256=sha(PARENT/'controls/result.json'),
        main_result_sha256=sha(ROOT/'main.json'),selection_order_sha256=__import__('hashlib').sha256(canonical(keys)).hexdigest(),
        no_method_reselection=True,no_weight_tuning=True,new_geometry_solves=0,
        human_quality_validation='NOT_EVALUABLE_NO_INDEPENDENT_RATINGS',can_route=False)
    write(out/'result.json',result)
    print('R65_FROZEN_SCORE_CONTROLS_DONE_NO_TRAJECTORY_OR_SELECTION_CHANGES',flush=True)
    return result


def verify():
    out = fresh(ROOT/'independent')
    main = read(ROOT/'main.json')
    if binding()['code'] != main['binding']['code']: raise ValueError('R65 scientific code changed')
    seed = read(ROOT/'private_seed.json')['seed']
    admin, manifest, preparation = build(out/'packet',seed)
    original = read(ROOT/'packet/reviewer_manifest.json')
    if manifest != original: raise ValueError('Reviewer media/HTML differ in independent rebuild')
    if sha(out/'packet/admin/mapping.json') != main['admin_mapping_sha256']:
        raise ValueError('Independent blind assignments differ')
    if review_analysis(admin,[]) != read(ROOT/'evaluation_empty.json'):
        raise ValueError('Empty review state changed')
    source = read(PARENT/'controls/result.json')
    from .diagnostics import batch
    keys = [v['record_key'] for c in admin['cases'] for v in c['variants']]
    controls_saved = read(ROOT/'controls/result.json')
    actual = batch(source['records'],source['cases'],source['music_clocks'],keys)
    if any(actual[k] != controls_saved[k] for k in actual):
        raise ValueError('Frozen proxy controls differ in independent replay')
    result = dict(binding=binding(),main_result_sha256=sha(ROOT/'main.json'),
        controls_result_sha256=sha(ROOT/'controls/result.json'),packet_files_bytes_exact=True,
        blind_assignments_exact=True,actual_trajectories_fk_recomputed=116,
        frozen_score_controls_exact=True,empty_human_analysis_exact=True,
        parent_files_unchanged=check_seal(main['input_seal']),
        human_records=0,human_reviewed=False,quality_conclusion='NOT_EVALUABLE',
        new_geometry_solves=0,neural_training_steps=0,optimizer_calls=0,Native24='NOT_EVALUABLE',can_route=False)
    write(out/'result.json',result)
    print('R65_INDEPENDENT_PACKET_AND_CONTROLS_EXACT',flush=True)
    return result


def evaluate(reviews, roster, out):
    out = fresh(out)
    docs = documents_from(reviews)
    admin = read(ROOT/'packet/admin/mapping.json')
    result = review_analysis(admin,docs,roster_from(roster))
    originals=[]
    for i,p in enumerate([] if reviews is None else sorted(Path(reviews).glob('*.json'))):
        dest=out/'private_original_records'/('%04d.json'%i);dest.parent.mkdir(exist_ok=True)
        shutil.copy2(str(p),str(dest))
        originals.append(dict(source_name=p.name,bytes=p.stat().st_size,sha256=sha(p),copy=dest.name))
    write(out/'private_original_records_manifest.json',dict(files=originals,byte_exact=True))
    write(out/'result.json',result)
    return result
