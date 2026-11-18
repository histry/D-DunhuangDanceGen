"""Rehash immutable R45 artifacts in a separate process after publication."""
import argparse,subprocess,datetime
from pathlib import Path
from dunhuang.research.complete_r36.core import read,write
from dunhuang.research.reperformance_r44.replay import digest

def run(out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh recheck receipt')
    records=[]
    for name in ('phase_r45_01','review_r45_01'):
        root=Path('outputs')/name;p=root/'artifacts_manifest.json';m=read(p);files=m['files']
        actual={f.relative_to(root).as_posix() for f in root.rglob('*') if f.is_file() and f.name not in ('artifacts_manifest.json','publication_receipt.json')}
        assert actual=={f['path'] for f in files}
        for f in files:
            item=root/f['path'];assert item.stat().st_size==f['bytes'] and digest(item)==f['sha256'],str(item)
        records.append(dict(root=str(root),files=len(files),bytes=sum(f['bytes'] for f in files),manifest_sha256=digest(p),source_commit=m['source_commit']))
    write(out,dict(records=records,all_hashes_verified=True,fresh_process=True,optimizer_calls=0,verified_by_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));print('R45_SEAL_RECHECK_OK',sum(r['files'] for r in records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
