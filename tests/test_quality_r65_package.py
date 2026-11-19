import copy
import pytest
from dunhuang.research.quality_r65.package import assignments,leak_check,opaque,public_order
from dunhuang.research.quality_r65.common import CFG

def population():
    seed='a1'*32
    cases=[dict(blind_item_id=opaque(seed,'case',str(i)),variants=[dict(method=m,blind_variant_id=opaque(seed,'variant',str(i)+m)) for m in CFG['methods']]) for i in range(29)]
    music=[dict(blind_item_id='TEST_FIXTURE_M'+str(i)) for i in range(3)]
    return seed,cases,music

def test_balanced_deterministic_opaque_assignments():
    seed,cases,music=population();forms=assignments(seed,cases,music)
    assert forms==assignments(seed,cases,music)
    assert forms!=assignments('a2'*32,cases,music)
    assert forms[0]['case_order']!=forms[1]['case_order']
    for f in forms:
        counts={m:[0]*4 for m in CFG['methods']}
        for c in cases:
            lookup={v['blind_variant_id']:v['method'] for v in c['variants']}
            assert set(f['panel_orders'][c['blind_item_id']])==set(lookup)
            for i,k in enumerate(f['panel_orders'][c['blind_item_id']]):counts[lookup[k]][i]+=1
        assert all(max(v)-min(v)<=1 and sum(v)==29 for v in counts.values())

@pytest.mark.parametrize('key',['method','source_uid','regions','metrics','role','time_warp','seed'])
def test_private_fields_rejected_even_if_nested(key):
    with pytest.raises(ValueError):leak_check({'cases':[{'variants':[{key:'TEST_FIXTURE'}]}]})

def test_method_value_rejected():
    with pytest.raises(ValueError):leak_check({'title':'LocalBeat'})
    assert leak_check({'data':'TEST_FIXTURE_COORDINATES','blind_item_id':'opaque'})

def test_public_array_order_does_not_retain_method_or_case_construction_order():
    rows=[{'blind_item_id':'z','variants':[{'blind_variant_id':k} for k in ('d','a','c','b')]},
          {'blind_item_id':'a','variants':[{'blind_variant_id':k} for k in ('c','b','d','a')]}]
    original=copy.deepcopy(rows);result=public_order(rows)
    assert rows==original
    assert [c['blind_item_id'] for c in result]==['a','z']
    assert all([v['blind_variant_id'] for v in c['variants']]==['a','b','c','d'] for c in result)
