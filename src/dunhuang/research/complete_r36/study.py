"""Build visible Module I effect before any additional training or ablation."""
import argparse,copy,subprocess
from pathlib import Path
import numpy as np
from .core import Library,read,write,sha
from .catalog import build
from .music import analyze
from .planning import search

def run(cfg,out,replay=None):
    out=Path(out)
    if out.exists():raise ValueError('Fresh output required')
    out.mkdir(parents=True)
    assert sha(cfg['catalog'])==cfg['catalog_sha256']
    lib=Library(cfg['catalog']);catalog=build(lib,cfg);write(out/'catalog.json',catalog)
    songs=[]
    for song in read(Path(cfg['r35'])/'planning/songs.json'):
        path=Path(song['audio']['path']);assert sha(path)==song['audio']['sha256']
        music=analyze(path,song['frames'],song['beats']);plan=search(catalog,music,lib,cfg)
        write(out/(song['id']+'_music.json'),music);write(out/(song['id']+'_plan.json'),plan)
        old=read(song['plan']['path']);before=len(old['events']);after=len(plan['events'])
        summary=dict(id=song['id'],name=song['name'],audio=song['audio'],frames=song['frames'],old_plan=song['plan'],
            new_plan=str(out/(song['id']+'_plan.json')),old_slots=before,new_units=after,
            mean_unit_frames=plan['frames']/after,family_counts={f:sum(e['family_uid']==f for e in plan['events']) for f in sorted({e['family_uid'] for e in plan['events']})},
            old_switches_per_min=(before-1)/(song['frames']/1800),new_switches_per_min=(after-1)/(song['frames']/1800))
        songs.append(summary);print('R36_PLAN',song['id'],before,after,flush=True)
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,
        source_catalog_sha256=sha(cfg['catalog']),units=len(catalog['units']),families=catalog['family_count'],
        source_counts={x['source_uid']:x['proposed_units'] for x in catalog['sources']},songs=songs,
        optimizer_steps=0,raw_deletions=0,old_baseline_changed=False,additional_ablation_calls=0,whole_song_generations=0,operational_qualified=False)
    write(out/'result.json',result)
    if replay:
        # Bind new paths, then compare all scientific data independently in a fresh process.
        for p in out.glob('*.json'):
            a=p.read_text(encoding='utf8').replace(str(out),'<RUN>');b=(Path(replay)/p.name).read_text(encoding='utf8').replace(str(Path(replay)),'<RUN>')
            assert a==b,p.name
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args()
    run(read('src/dunhuang/research/complete_r36/config.json'),a.out,a.replay_from)
