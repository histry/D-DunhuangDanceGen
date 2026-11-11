"""Bounded tmux-friendly source audit / feature smoke, with persistent receipts."""
import argparse,json,os,sys,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dunhuang.io import write
from dunhuang.data.sources import read,ref

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--raw-root',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--mode',choices=['audit','features'],default='audit');ap.add_argument('--audit')
    a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    assets=read(a.assets);env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
    if subprocess.check_output(['git','diff','HEAD','--'],cwd=ROOT,text=True):raise ValueError('Dirty tracked code')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    raw=Path(a.raw_root).resolve();start=time.monotonic()
    write(out/'protocol.json',dict(mode=a.mode,code_commit=commit,assets=ref(a.assets),
        population=ref(raw/'population.json'),raw_inventory=ref(raw/'inventory.json'),
        root_policy='both official global and fixed training XZ origin; preserve height; no floor fitting',
        frames=150,stride=75,normalizer_refitted=False,clipping=False,optimizer_steps=0,
        feature_policy='at most two distinct historical Train components; first representation-eligible windows, no quality ranking',
        old_roles_unchanged=True,blind_test_claimed=False,raw_assets_published=False))
    def command(args):
        print('RUN',args,flush=True);subprocess.run(list(map(str,args)),cwd=ROOT,env=env,check=True,timeout=3600)
    try:
        command([assets['edge_python'],'-m','pytest','-q'])
        if a.mode=='audit':
            command([assets['edge_python'],'scripts/verify_paired_conversion.py','--assets',a.assets,
                '--raw-root',raw,'--out',out/'conversion_parity.json'])
            command([assets['edge_python'],'-m','dunhuang.data.audit','--assets',a.assets,
                '--raw-root',raw,'--out',out/'audit'])
        else:
            if not a.audit:raise ValueError('Completed representation audit required')
            command([assets['edge_python'],'scripts/paired_feature_smoke.py','--assets',a.assets,
                '--raw-root',raw,'--audit',a.audit,'--out',out/'features'])
            command([assets['edge_python'],'scripts/train.py','--assets',out/'features/assets.json',
                '--task','paired','--preflight-only','--out',out/'paired_preflight'])
            report=read(out/'features/preflight.json');status=read(out/'paired_preflight/status.json')
            if report['ready'] or status['training_steps']!=0:raise ValueError('Uncalibrated assets unexpectedly admitted')
        write(out/'status.json',dict(status='COMPLETED_ASSET_RESEARCH_ONLY',mode=a.mode,exit_status=0,
            seconds=time.monotonic()-start,optimizer_steps=0,training_steps=0,
            music_qualified=False,full_motion_qualified=False))
    except Exception:
        write(out/'status.json',dict(status='FAILED',mode=a.mode,error=traceback.format_exc(),optimizer_steps=0));raise
if __name__=='__main__':main()
