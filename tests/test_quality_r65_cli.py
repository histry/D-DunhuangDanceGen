"""TEST_FIXTURE: CLI integration on synthetic dictionaries, not real reviews."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from dunhuang.research.quality_r65.metrics import summarize_reviews
from test_quality_r65_records import manifest, music_doc, checked_roster


@pytest.fixture
def cli(monkeypatch):
    # Only the CLI projection is under test. Do not import historical viewer or
    # source assets merely to monkeypatch its evaluate entry point.
    from dunhuang.research import quality_r65
    stub=ModuleType('dunhuang.research.quality_r65.study')
    for name in ('prepare','controls','verify','evaluate'):
        setattr(stub,name,lambda *args:pytest.fail('Unpatched TEST_FIXTURE entry point'))
    monkeypatch.setattr(quality_r65,'study',stub,raising=False)
    monkeypatch.setitem(sys.modules,stub.__name__,stub)
    path=Path(__file__).resolve().parents[1]/'scripts'/'quality_r65.py'
    spec=importlib.util.spec_from_file_location('TEST_FIXTURE_quality_r65_cli',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_result(out,result):
    out.mkdir()
    raw=(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf8')
    (out/'result.json').write_bytes(raw)
    return raw


def synthetic_alias_result(admin,doc,roster):
    alias=deepcopy(doc);alias['review_id']+='_ALIAS';alias['reviewer_id']='TEST_FIXTURE_ALIAS'
    roster=deepcopy(roster);roster['entries'].append(dict(roster['entries'][0],reviewer_id=alias['reviewer_id']))
    stats=summarize_reviews(admin,[doc,alias],roster)
    result=dict(schema='TEST_FIXTURE_PRIVATE_EVALUATION',stats=stats,
        automatic_vs_human=[dict(reviewer_id=d['reviewer_id'],kind='beat',metrics=dict(TP=1,FP=0,FN=0,F1=1.))
                           for d in (doc,alias)],
        actual_input_documents=2,actual_review_identities=2,original_documents_preserved=True)
    return result,doc,alias


def test_evaluate_cli_removes_alias_independence_without_deleting_originals(
        cli,manifest,music_doc,checked_roster,tmp_path,monkeypatch):
    result,first,alias=synthetic_alias_result(manifest,music_doc,checked_roster)
    assert result['stats']['independent_person_count']==1
    selected=[q['reviewer_id'] for q in result['stats']['qualifications'] if q.get('counted_as_independent')]
    out=tmp_path/'TEST_FIXTURE_output';saved={}
    def fake_evaluate(reviews,roster,folder):
        assert reviews=='TEST_FIXTURE_reviews' and roster=='TEST_FIXTURE_roster'
        saved['raw']=write_result(Path(folder),result)
        original=Path(folder)/'private_original_records';original.mkdir()
        saved['documents']={}
        for i,d in enumerate((first,alias)):
            raw=(json.dumps(d,indent=2)+'\n').encode()
            p=original/(str(i)+'.json');p.write_bytes(raw)
            saved['documents'][p]=raw
        return deepcopy(result)
    monkeypatch.setattr(cli.study,'evaluate',fake_evaluate)
    monkeypatch.setattr(sys,'argv',['quality_r65.py','evaluate','--reviews','TEST_FIXTURE_reviews',
        '--roster','TEST_FIXTURE_roster','--out',str(out)])
    cli.main()
    actual=json.loads((out/'result.json').read_text())
    assert [r['reviewer_id'] for r in actual['automatic_vs_human']]==selected
    assert len(actual['automatic_vs_human_nonindependent'])==1
    excluded=actual['automatic_vs_human_nonindependent'][0]
    assert excluded['reason']=='DUPLICATE_PERSON_DOCUMENT'
    assert 'EXCLUDED_FROM_INDEPENDENT' in excluded['scope']
    assert excluded['comparison'] in result['automatic_vs_human']
    assert actual['stats']==result['stats'] and actual['actual_input_documents']==2
    assert (out/'private_unfiltered_study_result.json').read_bytes()==saved['raw']
    meta=actual['independent_music_comparison_filter']
    assert meta['original_result_sha256']==hashlib.sha256(saved['raw']).hexdigest()
    assert meta['metrics_recomputed'] is False
    assert all(p.read_bytes()==raw for p,raw in saved['documents'].items())
    assert 'automatic_vs_human_nonindependent' not in result


def test_zero_label_evaluate_result_is_byte_unchanged(cli,manifest,tmp_path,monkeypatch):
    result=dict(stats=summarize_reviews(manifest,[]),automatic_vs_human=[],
                actual_input_documents=0,actual_review_identities=0)
    out=tmp_path/'TEST_FIXTURE_empty';saved={}
    def fake_evaluate(reviews,roster,folder):
        saved['raw']=write_result(Path(folder),result)
        return deepcopy(result)
    monkeypatch.setattr(cli.study,'evaluate',fake_evaluate)
    monkeypatch.setattr(sys,'argv',['quality_r65.py','evaluate','--out',str(out)])
    cli.main()
    assert (out/'result.json').read_bytes()==saved['raw']
    assert not (out/'private_unfiltered_study_result.json').exists()
    assert json.loads(saved['raw'])['stats']['status']=='NOT_EVALUABLE'


def test_only_explicit_counted_qualified_music_people_survive(cli,tmp_path):
    qualifications=[dict(schema='R65_MUSIC_REVIEW_V1',reviewer_id='GOOD',
                         contract_qualified=True,counted_as_independent=True),
        dict(schema='R65_QUALITY_REVIEW_V1',reviewer_id='QUALITY_ONLY',
             contract_qualified=True,counted_as_independent=True),
        dict(schema='R65_MUSIC_REVIEW_V1',reviewer_id='UNVERIFIED',
             contract_qualified=False,counted_as_independent=True),
        dict(schema='R65_MUSIC_REVIEW_V1',reviewer_id='NOT_COUNTED',contract_qualified=True),
        dict(schema='R65_MUSIC_REVIEW_V1',reviewer_id='DUPLICATE',
             contract_qualified=True,counted_as_independent=True,duplicate_person_document=True)]
    result=dict(stats=dict(qualifications=qualifications),automatic_vs_human=[
        dict(reviewer_id=q['reviewer_id'],kind='accent') for q in qualifications])
    out=tmp_path/'TEST_FIXTURE_scope';write_result(out,result)
    actual=cli.filter_independent_music_comparisons(result,out)
    assert [r['reviewer_id'] for r in actual['automatic_vs_human']]==['GOOD']
    assert len(actual['automatic_vs_human_nonindependent'])==4


@pytest.mark.parametrize('mode',['prepare','controls','verify'])
def test_non_evaluate_branches_never_invoke_filter(cli,mode,monkeypatch):
    calls=[]
    monkeypatch.setattr(cli.study,mode,lambda:calls.append(mode))
    monkeypatch.setattr(cli,'filter_independent_music_comparisons',
                        lambda *args:pytest.fail('Sealed execution branch must remain unchanged'))
    monkeypatch.setattr(sys,'argv',['quality_r65.py',mode]);cli.main()
    assert calls==[mode]


def test_mismatched_written_result_is_not_overwritten(cli,tmp_path):
    out=tmp_path/'TEST_FIXTURE_tamper';write_result(out,dict(other=True))
    raw=(out/'result.json').read_bytes()
    result=dict(stats=dict(qualifications=[]),automatic_vs_human=[dict(reviewer_id='MISSING')])
    with pytest.raises(ValueError):cli.filter_independent_music_comparisons(result,out)
    assert (out/'result.json').read_bytes()==raw
    assert not (out/'private_unfiltered_study_result.json').exists()
