"""Independent streamed rehash of the new R46 seal."""
import argparse,datetime,subprocess
from pathlib import Path
from dunhuang.research.complete_r36.core import read,write
from dunhuang.research.reperformance_r44.replay import digest

def run(out):
    out=Path(out);root=Path('outputs/bridge_audit_r46_01');p=root/'artifacts_manifest.json';m=read(p)
    if out.exists():raise ValueError('Fresh recheck receipt')
    actual={f.relative_to(root).as_posix() for f in root.rglob('*') if f.is_file() and f.name not in ('artifacts_manifest.json','publication_receipt.json')};assert actual=={f['path'] for f in m['files']}
    for f in m['files']:assert (root/f['path']).stat().st_size==f['bytes'] and digest(root/f['path'])==f['sha256'],f['path']
    write(out,dict(root=str(root),files=len(m['files']),bytes=sum(f['bytes'] for f in m['files']),manifest_sha256=digest(p),source_commit=m['source_commit'],verified_by_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_hashes_verified=True,fresh_process=True,optimizer_calls=0));print('R46_SEAL_RECHECK_OK',len(m['files']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
