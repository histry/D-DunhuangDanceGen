"""Three identity-selected real paired examples; never a quality demonstration."""
import json,subprocess
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import relative_fk
from .common import CFG,read,write,sha


def build(out,pop,rows,motions,clocks):
    out=Path(out);watch=out/'viewer';watch.mkdir()
    sk=read(CFG['data'])['skeleton'];J=np.asarray(sk['J']);parents=sk['parents']
    byid={r['uid']:r for r in rows};items=[]
    for case in pop['cases']:
        if case['key'] not in pop['display_case_keys']:continue
        row=byid[case['query_uid']];a,b=case['common_motion_interval']
        P=relative_fk(motions[case['source_uid']][0][a:b],J,parents)
        controls=[];clock=clocks[case['audio_sha256']]
        for delta in CFG['offsets']:
            name=case['key']+'_'+str(delta)+'.wav';start=(a+case['source_audio_offset_frames']+delta)/30
            subprocess.run(['ffmpeg','-nostdin','-v','error','-i',clock['audio_path'],'-ss',str(start),'-t',str((b-a)/30),
                '-vn','-c:a','pcm_s16le','-ac','1','-ar','22050',str(watch/name)],check=True)
            controls.append(dict(delta=delta,file=name,beat_frames=[t-a-case['source_audio_offset_frames']-delta for t in clock['beat_frames'] if a+case['source_audio_offset_frames']+delta<=t<b+case['source_audio_offset_frames']+delta]))
        items.append(dict(uid=case['query_uid'],source_uid=case['source_uid'],dataset=case['dataset'],role=case['role'],
            clock_scope=case['clock_scope'],interval=[a,b],P=np.round(P,5).tolist(),controls=controls,
            motion_peaks=[p['frame']-a for p in row['event_observations']],scores=row['scores']))
    data=json.dumps(dict(items=items,parents=parents),ensure_ascii=False,separators=(',',':'))
    template=Path(__file__).with_name('viewer.html').read_text(encoding='utf8')
    (watch/'index.html').write_text(template.replace('__DATA__',data),encoding='utf8')
    write(watch/'protocol.json',dict(scope='ACTUAL_PAIRED_MOTION_FIXED; AUDIO_CLOCK_ONLY_CHANGED; NOT_GENERATED_CHOREOGRAPHY',
        selection=pop['display_cases'],fps=30,frames=120,root_scope='BODY22_ROOT_RELATIVE; WORLD_ROOT_CONTACT_UNKNOWN',
        historical_exposure=True,human_reviewed=False,quality='NOT_EVALUABLE',music_offset_sign=CFG['offset_sign']))
    write(out/'viewer_manifest.json',dict(files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(watch.glob('*'))],
        selection_before_scores=True,no_raw_rotation_arrays=True,no_weights=True,no_full_audio=True))
