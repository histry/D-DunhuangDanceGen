import argparse,subprocess,datetime
from pathlib import Path
from dunhuang.research.complete_r36.core import read,write
from dunhuang.research.reperformance_r44.replay import digest
from .release import ROOTS

def run(out):
    if Path(out).exists():raise ValueError('Fresh independent seal receipt')
    records=[]
    for name in ROOTS:
        root=Path('outputs')/name;m=root/'artifacts_manifest.json';files=read(m)['files'];actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name not in ('artifacts_manifest.json','publication_receipt.json')};assert actual=={f['path'] for f in files}
        for f in files:assert (root/f['path']).stat().st_size==f['bytes'] and digest(root/f['path'])==f['sha256']
        records.append(dict(root=str(root),files=len(files),bytes=sum(f['bytes'] for f in files),manifest_sha256=digest(m)))
    write(out,dict(records=records,all_hashes_verified=True,fresh_process=True,neural_optimizer_calls=0,verified_by_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));print('R48_SEAL_OK',sum(r['files'] for r in records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
