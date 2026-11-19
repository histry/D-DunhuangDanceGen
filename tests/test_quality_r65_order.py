"""SYNTHETIC TEST FIXTURES ONLY; no actual reviewers, clocks or annotations."""
from copy import deepcopy

import pytest

from dunhuang.research.quality_r65.common import digest
from dunhuang.research.quality_r65.study import quality_order
from test_quality_r65_records import manifest, music_doc, quality_doc, checked_roster


@pytest.fixture
def ordered_three(manifest, music_doc, quality_doc, checked_roster):
    """Extend the shared record fixtures to the registered three-audio contract."""
    admin, music, quality, roster = map(deepcopy, (manifest, music_doc, quality_doc, checked_roster))
    for number in (2, 3):
        entry = deepcopy(admin['music'][0])
        entry['blind_item_id'] = 'M0'+str(number)
        admin['music'].append(entry)
        item = deepcopy(music['items'][0])
        item['blind_item_id'] = entry['blind_item_id']
        music['items'].append(item)
    music.update(review_id='TEST_FIXTURE_AUDIO_DOCUMENT', reviewed_at='2026-11-20T10:00:00+08:00')
    quality.update(review_id='TEST_FIXTURE_MOTION_DOCUMENT', reviewed_at='2026-11-20T11:00:00+08:00')
    roster['entries'][0].update(audio_before_motion_verified=True,
        music_review_id=music['review_id'], music_review_digest=digest(music),
        music_sealed_at='2026-11-20T10:01:00+08:00',
        motion_first_access_at='2026-11-20T10:02:00+08:00',
        order_evidence_path='TEST_FIXTURE_NOT_A_REAL_EVIDENCE_FILE',
        order_evidence_sha256='e'*64)
    return admin, music, quality, roster


def test_early_client_export_time_alone_never_grants_audio_first(ordered_three, monkeypatch):
    admin, music, quality, roster = ordered_three
    # Pure order checking must never read saved/real music clocks.
    import dunhuang.research.quality_r65.study as study
    monkeypatch.setattr(study, 'read', lambda *a, **k: pytest.fail('Pure quality_order read a file'))
    assert quality_order(quality, [music], admin, None) == (False, 'NO_VERIFIED_AUDIO_FIRST_ADMIN_RECORD')
    roster['entries'][0]['audio_before_motion_verified'] = False
    assert quality_order(quality, [music], admin, roster) == (False, 'AUDIO_FIRST_NOT_ADMIN_VERIFIED')


def test_changed_sealed_audio_document_digest_rejected(ordered_three):
    admin, music, quality, roster = ordered_three
    before = deepcopy((admin, music, quality, roster))
    roster['entries'][0]['music_review_digest'] = '0'*64
    assert quality_order(quality, [music], admin, roster) == (False, 'SEALED_AUDIO_DOCUMENT_CHANGED')
    assert (admin, music, quality) == before[:3]


def test_incomplete_three_music_population_rejected_even_with_matching_digest(ordered_three):
    admin, music, quality, roster = ordered_three
    music['items'].pop()
    roster['entries'][0]['music_review_digest'] = digest(music)
    assert quality_order(quality, [music], admin, roster) == (False, 'AUDIO_REVIEW_POPULATION_INCOMPLETE')


def test_correct_admin_seal_three_complete_music_and_order_accepted(ordered_three, monkeypatch):
    admin, music, quality, roster = ordered_three
    before = deepcopy((admin, music, quality, roster))
    import dunhuang.research.quality_r65.study as study
    monkeypatch.setattr(study, 'read', lambda *a, **k: pytest.fail('Pure quality_order read a file'))
    assert quality_order(quality, [music], admin, roster) == (True, None)
    assert (admin, music, quality, roster) == before


@pytest.mark.parametrize('sealed,opened', [
    ('2026-11-20T10:02:00+08:00', '2026-11-20T10:02:00+08:00'),
    ('2026-11-20T10:03:00+08:00', '2026-11-20T10:02:00+08:00'),
    ('2026-11-20T10:01:00', '2026-11-20T10:02:00+08:00'),
    ('2026-11-20T10:01:00+08:00', '2026-11-20T10:02:00'),
    ('not-a-time', '2026-11-20T10:02:00+08:00'),
])
def test_admin_seal_must_precede_first_motion_access_with_timezones(ordered_three, sealed, opened):
    admin, music, quality, roster = ordered_three
    roster['entries'][0].update(music_sealed_at=sealed, motion_first_access_at=opened)
    assert quality_order(quality, [music], admin, roster) == (False, 'ADMIN_AUDIO_SEAL_ACCESS_CLOCK_INVALID')


def test_partial_audio_coverage_or_task_is_not_full_music_completion(ordered_three):
    admin, music, quality, roster = ordered_three
    music['items'][0]['coverage_seconds'] = [0., 9.]
    roster['entries'][0]['music_review_digest'] = digest(music)
    assert quality_order(quality, [music], admin, roster) == (False, 'AUDIO_REVIEW_COVERAGE_INCOMPLETE')
    music['items'][0]['coverage_seconds'] = [0., 10.]
    music['items'][0]['tracks']['phrase']['status'] = 'PARTIAL'
    roster['entries'][0]['music_review_digest'] = digest(music)
    assert quality_order(quality, [music], admin, roster) == (False, 'AUDIO_REVIEW_TASK_UNFINISHED')


def test_exposed_or_wrong_assigned_audio_review_cannot_prove_independence(ordered_three):
    admin, music, quality, roster = ordered_three
    music['independence']['prior_r64_exposure'] = True
    roster['entries'][0]['music_review_digest'] = digest(music)
    assert quality_order(quality, [music], admin, roster) == (False, 'AUDIO_REVIEW_INDEPENDENCE_UNVERIFIED')
    music['form_id'] = 'F02'
    assert quality_order(quality, [music], admin, roster) == (False, 'SEALED_AUDIO_DOCUMENT_MISSING_OR_AMBIGUOUS')


def test_duplicate_audio_document_or_roster_identity_is_not_counted(ordered_three):
    admin, music, quality, roster = ordered_three
    assert quality_order(quality, [music, deepcopy(music)], admin, roster) == (
        False, 'SEALED_AUDIO_DOCUMENT_MISSING_OR_AMBIGUOUS')
    roster['entries'].append(deepcopy(roster['entries'][0]))
    assert quality_order(quality, [music], admin, roster) == (False, 'NO_VERIFIED_AUDIO_FIRST_ADMIN_RECORD')
