from dunhuang.research.judge_repair_r72.judge import parse

def test_short_schema_keeps_tie_and_unsure_distinct():
    assert parse('{"preference":"UNSURE"}','music')['preference']=='UNSURE'
    assert parse('{"preference":"TIE"}','music')['preference']=='TIE'
    assert parse('{"preference":"TIE","different":false}','naturalness')['different'] is False
    assert parse('{"preference":"A","different":"false"}','naturalness') is None

def test_localization_bounds_and_noncompliant_fields_not_filled():
    assert parse('{"seconds":4.0}','time')['seconds']==4
    assert parse('{"seconds":null}','time')['seconds'] is None
    assert parse('{"seconds":8}','time') is None
    assert parse('{"seconds":true}','time') is None
    assert parse('{"preference":"A","reason":"extra"}','music') is None
