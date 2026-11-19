"""TEST_FIXTURE only: rendering/contracts; never write actual human reviews."""
import base64
from copy import deepcopy
import json
import re
import shutil
import subprocess

import numpy as np
import pytest

from dunhuang.research.quality_r65 import viewer as v


@pytest.fixture
def packet():
    encoded = base64.b64encode(np.zeros((528, 22, 3), dtype='<f4').tobytes()).decode()
    cases = []
    for i in range(29):
        variants = [dict(blind_variant_id='TEST_FIXTURE_V%02d_%d' % (i, j),
                         data=encoded if i == j == 0 else None,
                         bounds=[[0, 0, 0], [0, 0, 0]] if i == j == 0 else None,
                         available=i == j == 0) for j in range(4)]
        cases.append(dict(blind_item_id='TEST_FIXTURE_C%02d' % i,
            audio=dict(path='audio/TEST_FIXTURE_clip.m4a', media_sha256='a'*64), variants=variants))
    music = [dict(blind_item_id='TEST_FIXTURE_M%d' % i,
        audio=dict(path='audio/TEST_FIXTURE_music%d.m4a' % i, media_sha256=str(i)*64),
        duration_seconds=120.+i) for i in range(3)]
    forms = []
    for f in range(2):
        ids = [c['blind_item_id'] for c in cases]
        order = ids if f == 0 else ids[::-1]
        panels = {c['blind_item_id']: [x['blind_variant_id'] for x in c['variants']][f:]
                  +[x['blind_variant_id'] for x in c['variants']][:f] for c in cases}
        forms.append(dict(form_id='F0%d' % (f+1), case_order=order, panel_orders=panels,
                          music_order=[m['blind_item_id'] for m in music][::1 if f == 0 else -1]))
    return dict(pack_id='TEST_FIXTURE_PACK', protocol_sha256='b'*64,
                fps=30, frames=528, parents=list(range(-1, 21)),
                dimensions=list(v.DIMENSIONS), forms=forms, cases=cases, music=music)


def embedded(path):
    text = path.read_text(encoding='utf8')
    data = re.search(r'<script id="r65-payload" type="application/json">(.*?)</script>', text, re.S)
    return text, json.loads(data.group(1))


def node(code):
    exe = shutil.which('node')
    if exe is None:
        pytest.skip('Node unavailable; pure JS export-contract tests are local verification')
    script = 'const assert=require("node:assert/strict");\n'+v.COMMON_JS+'\n'+code
    result = subprocess.run([exe, '-'], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_only_two_pages_written_coordinates_embedded_once_and_music_isolated(packet, tmp_path):
    files = v.write_viewers(tmp_path, packet)
    assert {p['path'] for p in files} == {'index.html', 'music.html'}
    assert {p.name for p in tmp_path.iterdir()} == {'index.html', 'music.html'}
    quality_html, quality = embedded(tmp_path/'index.html')
    music_html, music = embedded(tmp_path/'music.html')
    assert len(quality['cases']) == 29
    assert len(music['music']) == 3
    assert 'cases' not in music and 'parents' not in music and 'dimensions' not in music
    data = packet['cases'][0]['variants'][0]['data']
    assert quality_html.count(data) == 1 and data not in music_html
    assert all(set(f) == {'form_id', 'music_order'} for f in music['forms'])
    assert quality['forms'][1]['case_order'] == packet['forms'][1]['case_order']
    assert quality['forms'][1]['panel_orders'] == packet['forms'][1]['panel_orders']
    assert 'audio.currentTime*30' in quality_html
    assert 'grid-template-columns:repeat(2' in quality_html
    assert 'Math.random' not in quality_html.split('<script>')[-1]
    assert 'source_clock' not in quality_html and 'numerical_screen' not in quality_html
    assert 'music_clock' not in music_html and 'onset_strength' not in music_html
    assert 'media_sha256' in music_html


@pytest.mark.parametrize('where', ['root', 'case', 'variant', 'music', 'form'])
def test_admin_fields_cannot_leak_into_reviewer_packet(packet, tmp_path, where):
    target = {'root': packet, 'case': packet['cases'][0],
              'variant': packet['cases'][0]['variants'][0],
              'music': packet['music'][0], 'form': packet['forms'][0]}[where]
    target['admin_method_source_core_role_metrics_seed'] = 'TEST_FIXTURE_PRIVATE_CANARY'
    with pytest.raises(ValueError, match='whitelist'):
        v.write_viewers(tmp_path, packet)
    assert not (tmp_path/'index.html').exists()


def test_omitted_cases_changed_positions_and_remote_audio_rejected(packet):
    p = deepcopy(packet); p['cases'].pop()
    with pytest.raises(ValueError, match='All29'):
        v.project(p)
    p = deepcopy(packet); p['forms'][0]['panel_orders'][p['cases'][0]['blind_item_id']].pop()
    with pytest.raises(ValueError, match='four-panel'):
        v.project(p)
    p = deepcopy(packet); p['cases'][0]['audio']['path'] = 'https://example.org/upload.wav'
    with pytest.raises(ValueError, match='local audio'):
        v.project(p)
    p = deepcopy(packet); p['cases'][0]['variants'][1]['data'] = p['cases'][0]['variants'][0]['data']
    with pytest.raises(ValueError, match='blank'):
        v.project(p)


def test_quality_actual_touch_unsure_unrated_and_independence_contract_in_memory(packet):
    quality, _ = v.project(packet)
    node('const D='+json.dumps(quality)+r''';
const form=D.forms[1],id=form.case_order[0],vid=form.panel_orders[id][0];
const state={review_id:'TEST_FIXTURE_REVIEW',reviewer_id:'TEST_FIXTURE_PERSON',reviewer_roster_ref:'TEST_FIXTURE_ROSTER',independence:{own_judgment:true,no_other_reviews:true,no_method_identity:true,prior_r64_exposure:true},items:{}};
assert.throws(()=>r65BuildDocument('quality',D,form,state),/尚无/);
state.items[id]={touched:true,ratings:{[vid]:{music_coordination:'5',continuity:'UNSURE'}}};
const untouched=form.case_order[1];state.items[untouched]={touched:false,ratings:{}};
const doc=r65BuildDocument('quality',D,form,state);
assert.equal(doc.schema,'R65_QUALITY_REVIEW_V1');assert.equal(doc.items.length,1);
assert.deepEqual(doc.items[0].variant_order,form.panel_orders[id]);assert.equal(doc.items[0].ratings.length,4);
assert.deepEqual(doc.items[0].ratings[0].scores.music_coordination,{value:5,reason:null});
assert.deepEqual(doc.items[0].ratings[0].scores.continuity,{value:null,reason:'UNSURE'});
assert.deepEqual(doc.items[0].ratings[0].scores.richness,{value:null,reason:'UNRATED'});
assert.equal(doc.review_id,'TEST_FIXTURE_REVIEW');assert.equal(doc.independence.prior_r64_exposure,true);
assert.equal(doc.independence.other_reviews_seen,false);assert.ok(doc.reviewed_at.endsWith('Z'));
state.reviewer_id='';assert.throws(()=>r65BuildDocument('quality',D,form,state));state.reviewer_id='TEST_FIXTURE_PERSON';
state.independence.no_method_identity=false;assert.throws(()=>r65BuildDocument('quality',D,form,state));
''')


def test_music_defaults_partial_explicit_completion_and_no_clear_events_contract_in_memory(packet):
    _, music = v.project(packet)
    node('const D='+json.dumps(music)+r''';
const form=D.forms[0],id=form.music_order[0],tracks=Object.fromEntries(['beat','downbeat','accent','phrase'].map(k=>[k,{status:'PARTIAL',events:[]}]));
const state={review_id:'TEST_FIXTURE_REVIEW',reviewer_id:'TEST_FIXTURE_PERSON',reviewer_roster_ref:'TEST_FIXTURE_ROSTER',independence:{own_judgment:true,no_other_reviews:true,no_method_identity:true,prior_r64_exposure:false},items:{[id]:{touched:false,tracks,unsure_intervals:[]}}};
assert.throws(()=>r65BuildDocument('music',D,form,state),/尚无/);
state.items[id].touched=true;tracks.beat.events.push({seconds:12.345,confidence:'MEDIUM'});tracks.downbeat.status='UNSURE';
state.items[id].unsure_intervals.push({start:3,end:5,reason:'TEST_FIXTURE_REASON'});
let doc=r65BuildDocument('music',D,form,state),item=doc.items[0];
assert.equal(doc.schema,'R65_MUSIC_REVIEW_V1');assert.equal(item.tracks.beat.status,'PARTIAL');
assert.equal(item.tracks.downbeat.status,'UNSURE');assert.deepEqual(item.tracks.downbeat.events,[]);
assert.deepEqual(item.coverage_seconds,[0,120]);assert.equal(item.media_sha256,D.music[0].audio.media_sha256);
assert.deepEqual(item.tracks.beat.events,[{seconds:12.345,confidence:'MEDIUM'}]);
tracks.beat.status='NO_CLEAR_EVENTS';assert.throws(()=>r65BuildDocument('music',D,form,state),/空/);
tracks.beat.status='COMPLETE';doc=r65BuildDocument('music',D,form,state);assert.equal(doc.items[0].tracks.beat.status,'COMPLETE');
tracks.phrase.status='NO_CLEAR_EVENTS';assert.equal(r65BuildDocument('music',D,form,state).items[0].tracks.phrase.status,'NO_CLEAR_EVENTS');
state.items[id].unsure_intervals[0].end=121;assert.throws(()=>r65BuildDocument('music',D,form,state));
''')


def test_draft_refresh_keeps_review_identity_and_separates_tasks_in_memory():
    node(r'''
const memory=new Map();globalThis.localStorage={getItem:k=>memory.get(k)||null,setItem:(k,v)=>memory.set(k,v)};
const D={pack_id:'TEST_FIXTURE_PACK',protocol_sha256:'b'.repeat(64),forms:[{form_id:'F01'},{form_id:'F02'}]};
const store=r65ReadStore(D,'quality');store.drafts.F01={review_id:'TEST_FIXTURE_REVIEW',items:{}};
assert.equal(r65SaveStore(D,'quality',store),true);assert.equal(r65ReadStore(D,'quality').drafts.F01.review_id,'TEST_FIXTURE_REVIEW');
assert.deepEqual(r65ReadStore(D,'music').drafts,{});
assert.deepEqual(r65ReadStore({...D,protocol_sha256:'c'.repeat(64)},'quality').drafts,{});
''')
