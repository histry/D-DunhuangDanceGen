"""Independent streamed recheck of new R47 scientific roots."""
import argparse,datetime,subprocess
from pathlib import Path
from dunhuang.research.complete_r36.core import read,write
from dunhuang.research.reperformance_r44.replay import digest
from .release import ROOTS

def run(out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh seal receipt')
    records=[]
    for name in ROOTS:
        root=Path('outputs')/name;manifest=root/'artifacts_manifest.json';m=read(manifest);actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name not in ('artifacts_manifest.json','publication_receipt.json')};assert actual=={f['path'] for f in m['files']}
        for f in m['files']:assert (root/f['path']).stat().st_size==f['bytes'] and digest(root/f['path'])==f['sha256'],f['path']
        records.append(dict(root=str(root),files=len(m['files']),bytes=sum(f['bytes'] for f in m['files']),manifest_sha256=digest(manifest)))
    write(out,dict(records=records,all_hashes_verified=True,fresh_process=True,optimizer_calls=0,verified_by_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));print('R47_SEAL_RECHECK_OK',sum(r['files'] for r in records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
