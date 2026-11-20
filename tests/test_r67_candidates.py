"""TEST_FIXTURE: common-bank truth, reciprocal guards and exact DAG bounds."""
from copy import deepcopy

import numpy as np
import pytest

from dunhuang.research.sdmr_warp_rank import candidates as c


@pytest.fixture
def case():
    value = dict(key='TEST_FIXTURE_NEUTRAL_GRID', source_interval=[0, 240],
                 target_interval=[0, 240], support_component=[0, 240],
                 phase_inputs=[dict(type='synthetic_grid_anchor', frame=t) for t in (48, 96, 144, 192)],
                 beats=list(range(0, 240, 6)))
    value['time_warp'] = c.multi_warp(value['source_interval'], value['target_interval'],
                                     value['phase_inputs'], value['beats'])['time_warp']
    return value


def legal_injection(case, bank):
    # TEST_FIXTURE helper only. Production selects one hash ID and never retries.
    for number in range(100):
        result = c.inject(case, bank, 'TEST_FIXTURE_'+str(number))
        if result['status'] == 'CONSTRUCTED':
            return result
    raise AssertionError('Fixture contains no legal nonbaseline constructor')


def test_common_prefix_bank_has_exact_baseline_and_actual_distinct_q(case):
    bank = c.bank(case)
    assert bank == c.bank(case)
    assert len(bank['candidates']) == 16 and bank['legal_path_count'] >= 16
    assert bank['candidates'][0]['uid'] == bank['baseline_uid']
    assert bank['candidates'][0]['time_warp'] == case['time_warp']
    assert np.array_equal(bank['candidates'][0]['q'], np.arange(240))
    assert bank['design_anchor_scope'] == 'SYNTHETIC_GRID_NOT_MOTION_PEAKS_OR_MUSIC_BEATS'
    assert not bank['scores_consulted'] and bank['ranker_calls'] == 0
    ids = [row['uid'] for row in bank['candidates']]
    for k in c.KS:
        assert bank['prefixes'][str(k)] == ids[:k]
    for i, row in enumerate(bank['candidates']):
        assert .75-1e-10 <= row['source_to_target_rate_min'] <= row['source_to_target_rate_max'] <= 1.33+1e-10
        assert min(row['interpolation_read_support']) == 0 and max(row['interpolation_read_support']) == 239
        for old in bank['candidates'][:i]:
            assert not np.allclose(old['q'], row['q'], rtol=0, atol=1e-9)


def test_identical_q_duplicates_do_not_fill_a_scan_shortfall(case, monkeypatch):
    real = c.frozen.best
    def duplicate(graph, K):
        base = real(graph, 1)[0]
        # These are duplicate sampled clocks, not new candidate capacity.
        return [deepcopy(base) for _ in range(K)]
    monkeypatch.setattr(c.frozen, 'best', duplicate)
    bank = c.bank(case)
    assert len(bank['candidates']) == 1 and len(bank['prefixes']['16']) == 1
    assert bank['shortfall']['16'] == 15 and bank['artificial_duplicate_fill'] == 0
    assert bank['full_domain_unique_q_count'] is None
    assert bank['legal_path_count'] > 16


def test_baseline_mismatch_and_source_escape_rejected_before_scores(case):
    altered = deepcopy(case); altered['time_warp'][1][1] += 1
    with pytest.raises(ValueError, match='baseline'):
        c.bank(altered)
    altered = deepcopy(case); altered['support_component'] = [1, 240]
    with pytest.raises(ValueError, match='support'):
        c.bank(altered)
    for K in (0, True, 17, 4.):
        with pytest.raises(ValueError, match='budget'):
            c.bank(case, max_k=K)


def test_injected_inverse_is_a_preexisting_known_clock_not_an_appended_oracle(case):
    bank = c.bank(case)
    before = deepcopy(bank)
    injection = legal_injection(case, bank)
    assert bank == before
    truth = next(row for row in bank['candidates'] if row['uid'] == injection['known_candidate_uid'])
    assert np.allclose(injection['truth_q'], truth['q'], rtol=0, atol=1e-9)
    assert len(injection['q_injected']) == 240
    assert injection['q_injected'][0] == 0 and injection['q_injected'][-1] == 239
    assert injection['legality']['noncircular'] and injection['legality']['candidates_added'] == 0
    assert all(row['in_075_133'] for row in injection['legality']['rates'].values())
    errors = c.clock_errors(bank, injection['h'], bank['candidates'][0]['q'], [15, 225])
    known = next(row for row in errors if row['candidate_uid'] == injection['known_candidate_uid'])
    assert known['mae_frames'] < 1e-10 and known['inverse_q_mae_frames'] < 1e-10
    assert errors[0]['mae_frames'] > 0
    oracle = c.domain_oracle(bank, injection['h'], bank['candidates'][0]['q'], [15, 225])
    assert oracle['minimum_mae_frames'] < 1e-10
    assert oracle['bank_minimum_mae_frames'] < 1e-10
    assert not oracle['appended_to_bank']


def test_reciprocal_constructor_rejection_does_not_replace_selected_candidate(case):
    bank = c.bank(case, max_k=1)
    warp = [[0, 0], [48, 36], [96, 96], [144, 144], [192, 192], [239, 239]]
    row = c._candidate(case, [0, 240], [0, 240], [0, 240],
                       dict(time_warp=warp, cost=1.), 'TEST_FIXTURE_LEGAL_TAU')
    bank['candidates'].append(row)
    result = c.inject(case, bank, 'TEST_FIXTURE_FORCED_ONLY_NONBASELINE')
    assert result['status'] == 'NOT_CONSTRUCTIBLE'
    assert result['known_candidate_uid'] == row['uid']
    assert result['legality']['selected_candidate_remains_legal']
    assert result['legality']['rates']['injection_h']['in_075_133']
    assert not result['legality']['rates']['inverse_h']['in_075_133']
    assert result['legality']['rates']['inverse_h']['maximum'] == pytest.approx(4/3)
    assert not result['legality']['fallback_used'] and result['legality']['candidates_added'] == 0
    assert 'q_injected' not in result


def test_exact_DAG_oracle_matches_exhaustive_small_domain_and_detects_outside_truth(case):
    case = deepcopy(case)
    case['beats'] = list(range(0, 240, 12))
    case['time_warp'] = c.multi_warp([0, 240], [0, 240], case['phase_inputs'], case['beats'])['time_warp']
    bank = c.bank(case)
    h = [[0, 0], [60, 66], [120, 120], [180, 174], [239, 239]]
    baseline = np.asarray(bank['candidates'][0]['q'])
    oracle = c.domain_oracle(bank, h, baseline, [15, 225])
    count = bank['legal_path_count']
    assert count <= 100
    brute = []
    for rank in range(count):
        warp = np.asarray(c.frozen.unrank(bank['graph'], rank))
        q = np.interp(np.arange(240), warp[:, 1], warp[:, 0])
        brute.append(float(abs(np.interp(q, np.asarray(h)[:, 0], np.asarray(h)[:, 1])-baseline)[15:225].mean()))
    assert oracle['minimum_mae_frames'] == pytest.approx(min(brute), abs=1e-12)
    assert oracle['minimum_mae_frames'] > 0
    assert oracle['minimum_mae_frames'] <= oracle['bank_minimum_mae_frames']+1e-12
    assert oracle['status'] == 'EXACT_FULL_ORIGINAL_DAG_CLOCK_ORACLE'


def test_support_and_no_nonbaseline_remain_explicit(case):
    bank = c.bank(case, max_k=1)
    result = c.inject(case, bank, 'TEST_FIXTURE_ID')
    assert result['status'] == 'NOT_CONSTRUCTIBLE'
    assert result['known_candidate_uid'] is None
    with pytest.raises(ValueError, match='support'):
        c.clock_errors(bank, [[0, 0], [239, 239]], bank['candidates'][0]['q'], [20, 20])
    with pytest.raises(ValueError, match='endpoints'):
        c.domain_oracle(bank, [[1, 1], [239, 239]], bank['candidates'][0]['q'])
