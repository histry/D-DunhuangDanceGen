"""Truthful original-speed and audio-clock playback of saved body22 arrays."""
from pathlib import Path
import json,shutil,sys
import numpy as np
from .study import ROOT,load_study,CFG
from . import metrics
from dunhuang.research.local_timing_r68.common import read,write,sha
from dunhuang.research.sdmr_warp_rank import connectivity
from dunhuang.research.complete_r36.core import relative_fk

def compact(m):
 d=m['regions']['body'];return {k:d[k] for k in ['frames','low_cell_fraction','longest_low_seconds','low_runs','FK_D3_q95','FK_motion_range_rms_m','low_cells','cell_angular','cell_linear']}

def build(controls=False):
 root=ROOT/('controls' if controls else 'audit');out=root/'viewer';out.mkdir(exist_ok=False);ctx=connectivity.prepare();cases=read(ROOT/'audit/registration.json')['cases'];rows=read(ROOT/'audit/records.json');references={r['case_key']:r for r in read(ROOT/'audit/references.json')};natural=read(ROOT/'audit/natural_sources.json');items=[];cache={}
 packs={name:load_study(name) for name in ['R69','R73','GTW']};audio=read('outputs/retiming_baselines_r69_01/main/viewer/protocol.json')['audio_files']
 def positions(R):return np.round(relative_fk(R,ctx['J'],ctx['parents']),4).tolist()
 for ci,c in enumerate(cases):
  variants=[];uid=c['source_uid'];a,b=c['source_interval'];R=ctx['lib'].rotations[uid][a:b]
  requested=[('R69',m,None) for m in ['E1W','Uniform','Single64','Multi64']]+[('R73',m,None) for m in ['High_+6','Low_+6']]+[('GTW',m,6) for m in ['A_GTWLocal','A_GTWContext']]
  if controls:
   zc=np.load(root/'arrays.npz');controlrows=read(root/'records.json')
   requested=[]
   for r in controlrows:
    if r['case_key']!=c['key']:continue
    prefix=c['key']+'__'+r['method']+'__';present=prefix+'R' in zc
    variants.append(dict(label=r['method'],P=None if not present else positions(zc[prefix+'R']),fullP=None if prefix+'full_R' not in zc else positions(zc[prefix+'full_R']),q=None if not present else zc[prefix+'q'].tolist(),metrics=None if not present else compact(r['core']),fullMetrics=None if not r.get('full_pair') else compact(r['full_pair']['metrics']),status=r['status'],hold_cost=r.get('hold_cost'),music_hold_cost=r.get('music_hold_cost'),connection=(r.get('connection') or {}).get('passes_original_screen')))
   zc.close()
  for study,method,delta in requested:
   matched=next((r for r in rows if r['study']==study and r['case_key']==c['key'] and r['method']==method and (delta is None or r['unit'].endswith('_+6'))),None)
   label=study+' '+method
   if matched is None or matched['status']!='OBSERVED':variants.append(dict(label=label,P=None,fullP=None,metrics=None,status='MISSING_EXECUTION'));continue
   prefix=matched['unit']+'__'+method+'__';z=packs[study][3]
   variants.append(dict(label=label,P=positions(z[prefix+'R']),fullP=None if prefix+'full_R' not in z else positions(z[prefix+'full_R']),q=z[prefix+'q'].tolist(),metrics=compact(matched['core']),fullMetrics=None if not matched['full_pair'] else compact(matched['full_pair']['metrics']),status='OBSERVED',connection=matched['geometry_screen']))
  clip=next(v for v in audio.values() if v['raw_audio_sha256']==c['audio_sha256'] and v['pair_music_interval']==c['pair_music_interval']);dest=out/clip['path'];dest.parent.mkdir(exist_ok=True);src=Path('outputs/retiming_baselines_r69_01/main/viewer')/clip['path'];assert sha(src)==clip['sha256'];shutil.copyfile(src,dest)
  items.append(dict(label=f"{ci+1}. {uid} [{a},{b})",case_key=c['key'],source_uid=uid,source_interval=[a,b],sourceP=positions(R),sourceMetrics=compact(references[c['key']]['natural']),variants=variants,audio=clip['path'],core_offset=0 if c['parent_side']=='left' else 288,source_holds=references[c['key']]['source_holds']))
 for p in packs.values():p[3].close()
 template=Path(__file__).with_name('viewer.html').read_text(encoding='utf8')
 payload=dict(title='停顿控制机制先导' if controls else '来源／E1-W／GTW · 低活动审计',parents=ctx['parents'].tolist(),items=items,manifest_sha256=sha(root/'manifest.json'),human_records=0)
 (out/'index.html').write_bytes(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':'))).encode())
 # Separate original full recordings: no music and no time normalization.
 full=[]
 for uid in sorted({c['source_uid'] for c in cases}):
  R=ctx['lib'].rotations[uid];full.append(dict(label=uid,source_uid=uid,source_interval=[0,len(R)],sourceP=positions(R),sourceMetrics=compact(dict(regions=natural[uid]['regions'])),variants=[],audio=None,core_offset=0,source_holds=[]))
 (out/'source.html').write_bytes(template.replace('__DATA__',json.dumps(dict(payload,title='完整来源 · 原速独立播放',items=full),ensure_ascii=False,separators=(',',':'))).encode())
 write(out/'protocol.json',dict(source_manifest_sha256=payload['manifest_sha256'],cases=len(items),full_original_recordings=len(full),original_fps=30,source_clock='INDEPENDENT_REAL_TIME_NEVER_TARGET_NORMALIZED',target_clock='ORIGINAL_AUDIO',body='ROOT_RELATIVE_22_NO_FINGERS_WORLD_ROOT_OR_CONTACT',semantic_hold='UNKNOWN',human_records=0,can_route=False))
 write(root/'viewer_delivery.json',dict(files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]));print('HOLD_VIEW_READY',len(items),len(full),flush=True)

if __name__=='__main__':build('--controls' in sys.argv)
