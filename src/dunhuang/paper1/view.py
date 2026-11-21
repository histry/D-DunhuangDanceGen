"""Actual complete-song and candidate playback, all clocks explicitly labelled."""
import base64,gzip,json,shutil
import numpy as np
from dunhuang.research.complete_r36.core import relative_fk
from .contracts import fresh,read,write,sha

def encoded(P):return base64.b64encode(np.asarray(P,dtype='<f4').tobytes()).decode()

def build_view(out,pop,arrays,result):
    folder=fresh(out/'viewer');songs=[];sources={}
    for s in result['songs']:
        item=dict(song=s['song'],frames=s['frames'],regions=s['regions'],audio=f"song{s['song']}.wav",audio_sha256=s['audio']['audio_sha256'],
                  baseline=encoded(arrays[f"song{s['song']}_baseline_P"]),slots=[])
        for slot in s['slots']:
            rows=[]
            for c in slot['candidates']:
                unit=c['event_uid'];a,b=c['source_interval']
                if unit not in sources:
                    sources[unit]=dict(source_uid=c['source_uid'],source_interval=[a,b],frames=b-a,
                                      data=encoded(relative_fk(pop.lib.rotations[c['source_uid']][a:b],pop.net.joints,pop.net.parents)))
                rows.append(dict(c,data=encoded(relative_fk(arrays[c['uid']],pop.net.joints,pop.net.parents)),q=arrays[c['uid']+'_q'].tolist()))
            item['slots'].append(dict(slot,candidates=rows))
        songs.append(item)
    payload=dict(songs=songs,sources=sources,parents=pop.net.parents.tolist(),scope='P0_READ_ONLY_FROZEN_BASELINE_AND_FINITE_EXECUTED_CLOCKS')
    raw=json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode()
    (folder/'data.js').write_text('window.P0_GZIP="'+base64.b64encode(gzip.compress(raw,mtime=0)).decode()+'";',encoding='utf8')
    shutil.copyfile(__import__('pathlib').Path(__file__).with_name('viewer.html'),folder/'index.html')
    write(folder/'protocol.json',dict(songs=3,slots=sum(len(s['slots']) for s in songs),candidates=sum(len(k['candidates']) for s in songs for k in s['slots']),
          source_records=len(sources),audio_transferred=False,existing_audio=[dict(name=s['audio'],sha256=s['audio_sha256']) for s in songs],
          body_scope='ACTUAL_BODY22_ROOT_RELATIVE; NOT_WORLD_ROOT_CONTACT',baseline='EXACT_R56_B0_NOT_NEW_PLANNER_OUTPUT',
          files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.iterdir())},can_route=False))
