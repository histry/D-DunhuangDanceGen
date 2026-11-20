"""Actual R73 trajectories and exact shared audio clock; no synthetic labels."""
import json,shutil,sys
from pathlib import Path
import numpy as np
from .study import ROOT,PARENT
from dunhuang.research.local_timing_r68.common import read,write,sha
from dunhuang.research.sdmr_warp_rank import connectivity
from dunhuang.research.complete_r36.core import relative_fk

def build(mode='preview'):
    source=ROOT/mode;out=source/'viewer';out.mkdir(exist_ok=False);(out/'audio').mkdir()
    cases=read(source/'registration.json')['cases'];groups=read(source/'groups.json');records=read(source/'records.json')
    z=np.load(source/'arrays.npz');ctx=connectivity.prepare();audio_files=read(PARENT/'viewer/protocol.json')['audio_files'];payload=[]
    for group in groups:
        case=cases[group['case_index']];key=case['key'];variants=[]
        for method in ['E1W']+group['methods']:
            row=next(r for r in records if r['case_key']==key and r['method']==method)
            prefix=key+'__'+method+'__';full=z[prefix+'full_R'] if prefix+'full_R' in z else None
            core=z[prefix+'R'];q=z[prefix+'q'];comp=z[prefix+'compensation_mask'].tolist() if prefix+'compensation_mask' in z else [False]*240
            variants.append(dict(method=method,P=None if full is None else np.round(relative_fk(full,ctx['J'],ctx['parents']),4).tolist(),
                core_P=np.round(relative_fk(core,ctx['J'],ctx['parents']),4).tolist(),q=q.tolist(),compensation_mask=comp,
                G1=row['connection']['passes_original_screen'],match=row.get('match'),
                actual_peak=row.get('candidate',{}).get('actual_peak_frame'),mapped_anchor=row.get('candidate',{}).get('mapped_anchor_target_frame')))
        audio=next(p for p in audio_files.values() if p['raw_audio_sha256']==case['audio_sha256'] and p['pair_music_interval']==case['pair_music_interval'])
        original=PARENT/'viewer'/audio['path'];assert sha(original)==audio['sha256'];shutil.copyfile(original,out/audio['path'])
        a,b=case['source_interval'];native=ctx['lib'].rotations[case['source_uid']][a:b]
        payload.append(dict(case_index=group['case_index'],source_uid=case['source_uid'],source_interval=case['source_interval'],delta=group['delta'],status=group['status'],
            primary_match=group['primary_match'],audio=audio['path'],core_start=0 if case['parent_side']=='left' else 288,variants=variants,
            native=np.round(relative_fk(native,ctx['J'],ctx['parents']),4).tolist()))
    data=dict(cases=payload,parents=ctx['parents'].tolist(),source_manifest_sha256=sha(source/'manifest.json'))
    template=Path(__file__).with_name('viewer.html').read_text(encoding='utf-8')
    (out/'index.html').write_bytes(template.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':'))).encode('utf-8'))
    write(out/'protocol.json',dict(source_manifest_sha256=sha(source/'manifest.json'),frames=528,fps=30,core_frames=240,bridge_frames=48,
        native_view='SEPARATE_SILENT_ORIGINAL30FPS_NO_RETIMING',primary='MATCHED_HIGH_LOW_ONLY',distributed='MATCH_GATES_REPORTED_SEPARATELY',human_reviewed=False,Native24='NOT_EVALUABLE'))
    write(source/'viewer_delivery.json',dict(files=[dict(path=p.relative_to(source).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()]))
    print('R73_VIEW_READY',mode,len(payload),flush=True)

if __name__=='__main__':build('controls' if '--controls' in sys.argv else 'preview')
