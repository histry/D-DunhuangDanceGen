"""Train-only necessary screen, plus distinct 22-joint versus full Native scopes."""
import argparse,collections
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write
from dunhuang.research.sdmr_event_rank.evaluate import group_ci

def run(main,controls,out):
    a=read(Path(main)/'result.json');b=read(Path(controls)/'result.json');lookup={r['key']:r for r in b['records']};rows=[];paired=collections.defaultdict(lambda:dict(WrongMapB=[],NoMusicTime=[]));change=[]
    for r in a['records']:
        key=r['key'];m=lookup[key]['metrics']['methods'];g=m['Geometry'];c=m['C2'];ratio=c['core_activity']/max(g['core_activity'],1e-12);d3=c['core_D3']/max(g['core_D3'],1e-12);valid=.85<=ratio<=1.15 and d3<=1.05 and c['protected_exact'];source=r['source'];rows.append(dict(key=key,domain=source['domain'],role=source['role'],activity_ratio=ratio,core_D3_ratio=d3,protected_exact=c['protected_exact'],dynamics_screen=valid,metrics=m))
        if source['domain']=='ChangE' and source['role']=='train':change.append(valid)
        if source['domain']=='AIST++' and source['role']=='train':
            group=source['music_identity']['source']['sha256'];v=c['rms_accent_timing_proxy_seconds']
            for control in ('WrongMapB','NoMusicTime'):
                other=m[control]['rms_accent_timing_proxy_seconds']
                if v is not None and other is not None:paired[group][control].append(other-v)
    comparisons={}
    for control in ('WrongMapB','NoMusicTime'):
        groups={g:float(np.mean(v[control])) for g,v in paired.items() if v[control]};ci=group_ci(list(groups.values()),44041,5000);comparisons[control]=dict(group_mean_improvement_seconds=groups,ci95=ci,passed=ci[0] is not None and ci[0]>0)
    screen=sum(change)>=3 and all(v['passed'] for v in comparisons.values());result=dict(records=rows,train_only_screen=dict(change_passed=sum(change),change_total=len(change),music_proxy=comparisons,passed=screen),neural_training_admitted=False if not screen else None,training_decision='STOP_NO_CREDIBLE_TRAIN_ONLY_BENEFIT' if not screen else 'NEEDS_REGISTERED_REAL_PAIRED_NEURAL_POSITIVE_CONTROL',full_native24='NOT_EVALUABLE_22_JOINTS_UNKNOWN_WORLD_ROOT_CONTACT',semantic_music_alignment='NOT_ESTABLISHED',optimizer_steps=0,whole_song_generated=False,can_route=False,operational_qualified=False);write(out,result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--controls',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.main,a.controls,a.out)
