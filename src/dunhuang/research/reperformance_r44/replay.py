"""Fresh-process replay; parent content verification uses streaming hashes."""
import argparse,hashlib
from pathlib import Path
from dunhuang.research.complete_r36.core import read,write
from dunhuang.research.music_reperformance_r43.replay import exact_arrays

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def compare(first,second,out):
    first=Path(first);second=Path(second);a=read(first/'result.json');b=read(second/'result.json');a.pop('code_commit',None);b.pop('code_commit',None);assert a==b;exact_arrays(first/'arrays.npz',second/'arrays.npz');write(out,dict(exact_arrays=True,exact_scientific_json=True,first_arrays_sha256=digest(first/'arrays.npz'),replay_arrays_sha256=digest(second/'arrays.npz'),optimizer_calls=0))

def parents(out):
    records=[]
    for name in ('bridge_r38_01','stabilize_r39_01','contract_r40_01','paired_event_r41_01','source_context_r42_01','music_reperformance_r43_01','sdmr_event_rank_01'):
        root=Path('outputs')/name;manifest=root/'artifacts_manifest.json';n=0;size=0
        for r in read(manifest)['files']:
            p=root/r['path'];assert p.is_file() and p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],str(p);n+=1;size+=r['bytes']
        records.append(dict(root=str(root),manifest_sha256=digest(manifest),verified_files=n,verified_bytes=size))
    write(out,dict(parents=records,all_parent_hashes_unchanged=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--first');p.add_argument('--second');p.add_argument('--out',required=True);p.add_argument('--parents',action='store_true');a=p.parse_args();parents(a.out) if a.parents else compare(a.first,a.second,a.out)
