"""Render only actual selected arrays; original speed is a separate clock."""
from pathlib import Path
import json,subprocess,sys
import numpy as np
from scipy.spatial.transform import Rotation
from .study import ROOT,HERE,read,write,sha,prepare
from dunhuang.research.complete_r36.core import relative_fk

def build(phase='main'):
    root=ROOT/phase;out=root/'viewer';out.mkdir(exist_ok=False);ctx=prepare();reg=read(root/'registration.json');records=read(root/'records.json');z=np.load(root/'arrays.npz',allow_pickle=False);items=[]
    def P(R):return np.round(relative_fk(R,ctx['J'],ctx['parents']),4).tolist()
    for c in reg['cases']:
        a,b=c['unit']['source_interval'];s=c['unit']['source_uid'];T=c['target_frames'];variants=[]
        for r in records:
            if r['case_key']!=c['key']:continue
            n=c['key']+'__'+r['method'];variants.append(dict(label=r['method'],P=P(z[n+'__R']),fullP=P(z[n+'__full_R']) if n+'__full_R' in z else None,q=z[n+'__q'].tolist(),metrics={k:r['core'][k] for k in ['low_cells','longest_low_seconds','low_cell_fraction','FK_D3_q95','FK_activity_path_mean_m']},source_interval=r['source_interval'],connection=r['audit']['bridge_screen'],source_uid=r['source_uid']))
        audio='music_'+str(c['music_song'])+'_'+str(T)+'.wav';dest=out/audio
        if not dest.exists():
            source=Path('outputs/complete_r36_01/viewer/audio')/('dunhuangwu%d.wav'%(c['music_song']+2));subprocess.run(['ffmpeg','-v','error','-i',str(source),'-ss',str(c['music_start_frame']/30),'-t',str((2*T+48)/30),'-acodec','pcm_s16le',str(dest)],check=True)
        items.append(dict(label=s+' '+str([a,b])+' / '+str(T)+' frames',key=c['key'],sourceP=P(ctx['lib'].rotations[s][a:b]),source_interval=[a,b],hold=c['hold']['cells_half_open'],T=T,audio=audio,variants=variants,alternative=c['alternative']['source_interval'] if c['alternative'] else None))
    data=dict(title='Paper 1 · 来源容量与执行时钟'+(' · 对照' if phase=='controls' else ' · 主候选'),parents=ctx['parents'].tolist(),items=items,inventory=reg['source_inventory'],status=reg['status'])
    template=(HERE/'viewer.html').read_text(encoding='utf8');(out/'index.html').write_bytes(template.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':'))).encode())
    # Raw multi-person tracks are inspection only. Twelve seconds, no time compression.
    raw=[]
    for s in reg['source_inventory']['rows']:
        with np.load(ROOT/'assets/change_official_14'/(s['source_uid']+'.npz'),allow_pickle=False) as rawz:
            R=Rotation.from_rotvec(rawz['poses'][:360,:66].reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
        raw.append(dict(label=s['source_uid']+' / '+s['admission'],key=s['source_uid'],sourceP=P(R),source_interval=[0,len(R)],T=len(R),audio=None,variants=[],hold=None,alternative=None))
    source=dict(data,title='14条原始轨道 · 首12秒原速检查（非整条缩时）',items=raw)
    (out/'sources.html').write_bytes(template.replace('__DATA__',json.dumps(source,ensure_ascii=False,separators=(',',':'))).encode())
    write(out/'protocol.json',dict(phase=phase,manifest_sha256=sha(root/'manifest.json'),source_clock='INDEPENDENT_30FPS_NATURAL_SPEED',target_clock='SINGLE_AUDIO_MASTER_FRAME_ROUNDING_WITH_PAUSE_STEP_SEEK',raw_track_preview='FIRST_360_REAL_FRAMES_NOT_FULL_RECORDING',body='ROOT_RELATIVE22_Y_UP_METERS_NO_FINGERS_OR_CONTACT',human_records=0,quality='NOT_EVALUABLE',can_route=False,files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()}))
    z.close();print('PAPER1_SUPPORT_VIEW_READY',phase,len(items),flush=True)
if __name__=='__main__':build(sys.argv[1] if len(sys.argv)>1 else 'main')
