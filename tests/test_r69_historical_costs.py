"""Synthetic accounting fixtures, never measurements from real R69 results."""
import copy
import importlib.util
from pathlib import Path

import pytest

path=Path(__file__).with_name('historical_costs.py')
if not path.is_file():path=Path(__file__).resolve().parents[1]/'scripts/historical_costs_r69.py'
spec=importlib.util.spec_from_file_location('r69_historical_costs_tested',path)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def fixture():
    banks=[dict(case_key='a',bank=dict(candidates=[dict(uid='u'),dict(uid='v')])),
        dict(case_key='b',bank=dict(candidates=[dict(uid='w')]))]
    records=[dict(case_key=case,candidate_uid=uid,connectivity=dict(
        actual_geometry_calls_this_request=actual,cached_geometry_reuses_this_request=1-actual,
        cache_key='synthetic_'+uid)) for case,uid,actual in [('a','u',1),('a','v',0),('b','w',1)]]
    registration=dict(connectivity_counts=dict(check_requests=3,actual_geometry_calls=2,cached_geometry_reuses=1))
    return banks,records,registration


def test_shared_bank_counted_once_with_real_receipt_cache_accounting():
    args=fixture();out=h.shared_historical_costs(*args)
    assert out['totals']['motion_encoding_calls_inferred']==3
    assert out['totals']['music_encoding_calls_inferred']==2
    assert out['totals']['actual_geometry_calls']['total']==2
    assert out['totals']['cached_geometry_reuses']['total']==1
    assert out['registration_receipt_agreement']==dict(check_requests=True,actual_geometry_calls=True,cached_geometry_reuses=True)
    assert out['historical_wall_seconds'] is None
    assert 'NOT_INSTRUMENTED' in out['encoding_count_basis']
    assert 'NO_SEPARATE_FULL_BANK' in out['E1W_scope']
    assert h.shared_historical_costs(*args)==out


def test_missing_actual_receipt_not_replaced_by_global_counter_or_zero():
    banks,records,registration=fixture();records[0]['connectivity']={}
    out=h.shared_historical_costs(banks,records,registration)
    assert out['totals']['actual_geometry_calls']==dict(total=None,known_sum=1,known_requests=2,missing_requests=1)
    assert out['registration_counts']['actual_geometry_calls']==2
    assert out['registration_receipt_agreement']['actual_geometry_calls'] is None
    assert out['banks'][0]['candidate_receipts'][0]['actual_geometry_calls'] is None
    without=h.shared_historical_costs(banks,records,{})
    assert all(value is None for value in without['registration_counts'].values())


@pytest.mark.parametrize('problem',['duplicate_bank','duplicate_row','orphan_row','missing_row','counter_conflict','boolean_count'])
def test_accounting_identity_or_receipt_errors_are_not_silently_repaired(problem):
    banks,records,registration=copy.deepcopy(fixture())
    if problem=='duplicate_bank':banks.append(copy.deepcopy(banks[0]))
    elif problem=='duplicate_row':records.append(copy.deepcopy(records[0]))
    elif problem=='orphan_row':records.append(dict(case_key='x',candidate_uid='orphan'))
    elif problem=='missing_row':records.pop()
    elif problem=='counter_conflict':registration['connectivity_counts']['actual_geometry_calls']=3
    else:records[0]['connectivity']['actual_geometry_calls_this_request']=True
    with pytest.raises(ValueError):h.shared_historical_costs(banks,records,registration)
