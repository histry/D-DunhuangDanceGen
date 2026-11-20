"""Read-only score attachment to saved R69/GTW motion, never quality ranking."""
from pathlib import Path
import numpy as np
from .infer import ROOT
from .catalog import reward
from dunhuang.research.local_timing_r68.common import read,write,sha
from dunhuang.research.paired_event_r41.events import profiles,phases
from dunhuang.research.warp_coupling_r53.common import array_hash

def run():
    clocks={}
    for p in (ROOT/'main').glob('*/catalog.json'):
        c=read(p);clocks[c['audio_sha256']]=c
    cfg=read('src/dunhuang/research/manifold_bridge_r47/config.json');rows=[];bindings={};cache={}
    for label,parent,expected in [('R69',Path('outputs/retiming_baselines_r69_01/main'),'57ea35f5d7bd1d5d63da065e46e59951698d78b05c844de9aae711f890b08abe'),('GTW',Path('outputs/gtw_baseline_01/controls'),'9f9419779c1a9335b9c55f6536db691e1873a95be17d4d0f2a3861300785454c')]:
        assert sha(parent/'manifest.json')==expected;bindings[label]=expected
        cases=read(parent/'population.json')['cases'] if label=='R69' else read(parent/'registration.json')['cases']
        bykey={c['key']:c for c in cases};z=np.load(parent/'arrays.npz');records=read(parent/'records.json')
        for r in records:
            case=bykey[r['case_key']] if label=='R69' else cases[r.get('case_index',0)]
            unit=r['case_key'] if label=='R69' else r['unit'];name=unit+'__'+r['method']+'__full_R'
            row=dict(study=label,unit=unit,method=r['method'],audio_sha256=case['audio_sha256'],status='MISSING_EXECUTION',quality='NOT_EVALUABLE',Native24='NOT_EVALUABLE')
            if name not in z:rows.append(row);continue
            R=z[name];h=array_hash(R)
            if h not in cache:
                cache[h]=[e['frame']/30 for e in phases(profiles(R),cfg) if e['body_region']=='upper' and e['type']=='speed_peak']
            start=case['pair_music_interval'][0]/30;p=np.asarray(cache[h])+start;c=clocks[case['audio_sha256']];scores={}
            for model,kind in [('beat_this','beat'),('beat_this','downbeat'),('allin1','beat'),('allin1','downbeat'),('acoustic','accent_candidate'),('R64','beat'),('R64','accent_candidate')]:
                e=[v['seconds'] for v in c['events'] if v['source']==model and v['kind']==kind and v['seconds_supported']]
                scores[model+'_'+kind]=reward(p,e)
            rows.append(dict(row,status='OBSERVED',actual_motion_sha256=h,output_frames=len(R),motion_event_count=len(p),scores=scores))
    out=ROOT/'saved_motion_diagnostics';out.mkdir(exist_ok=False);write(out/'scores.json',rows)
    write(out/'qualification.json',dict(rows=len(rows),actual=sum(r['status']=='OBSERVED' for r in rows),unique_motion_profiles=len(cache),source_bindings=bindings,
        metric='R47_UPPER_SPEED_PEAK_GAUSSIAN_SIGMA0.1S_NOT_ORIGINAL_BAS_IMPLEMENTATION',quality_winner=None,quality='NOT_EVALUABLE',can_route=False,optimizer_calls=0,training_steps=0,novel_generated_motion=0))

if __name__=='__main__':run()
