"""Derive a display-only correction without rewriting frozen review/scoring inputs."""
import argparse, shutil
from pathlib import Path
from dunhuang.research.complete_r36.core import read,write,sha

def run(package,out):
    package=Path(package);out=Path(out);source=package/'reviewer'
    if out.exists():raise ValueError('Fresh display directory required')
    original=read(source/'protocol.json')
    for f in original['files']:
        assert Path(f['name']).name==f['name']
        assert sha(source/f['name'])==f['sha256']
    shutil.copytree(source,out)
    shutil.copyfile(Path(__file__).with_name('style.css'),out/'style.css')
    for f in original['files']:
        if f['name']!='style.css':assert sha(out/f['name'])==f['sha256']
    write(out/'protocol.json',dict(original,source_protocol_sha256=sha(source/'protocol.json'),admin_manifest_sha256=sha(package/'admin/manifest.json'),display_only_change='HONOR_HIDDEN_AUDIO_IN_EVENT_POSE; NO_MEDIA_LABEL_OR_SCORE_CHANGE',files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.name!='protocol.json']))
    print('R45_REVIEW_DISPLAY_ONLY',len(original['files']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.package,a.out)
