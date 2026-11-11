"""Persistent stage runner. Never starts extraction/training before its input gate."""
import argparse,sys,subprocess,os,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dunhuang.io import write
from dunhuang.data.sources import read,ref
from dunhuang.data.aist_pairs import run_calibration,compile_index,extract_features

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['calibrate','index','features','train'],required=True)
    ap.add_argument('--assets',required=True);ap.add_argument('--raw-root',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--authority');ap.add_argument('--source-audio');ap.add_argument('--calibration');ap.add_argument('--audit')
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--index');ap.add_argument('--steps',type=int,default=300);ap.add_argument('--workers',type=int,default=6)
    a=ap.parse_args();out=Path(a.out).resolve();assets=read(a.assets)
    if out.exists():raise FileExistsError(out)
    if subprocess.check_output(['git','diff','HEAD','--'],cwd=ROOT,text=True):raise ValueError('Uncommitted tracked code')
    begun=time.monotonic()
    try:
        if a.stage=='calibrate':
            if not 1<=a.workers<=8:raise ValueError('Bounded parallel video jobs required')
            run_calibration(assets,a.raw_root,a.authority,out,a.source_audio,a.workers,a.smoke)
        elif a.stage=='index':compile_index(assets,a.raw_root,a.calibration,a.audit,out)
        elif a.stage=='features':extract_features(assets,a.index,out)
        else:
            from dunhuang.data.paired_manifest import inspect
            _,_,preflight=inspect(assets)
            if not preflight['ready']:raise ValueError('Complete paired preflight blocked')
            from dunhuang.generation.train import train
            train(assets,out,a.steps,task='paired',preflight_only=False)
        write(out/'stage_exit.json',dict(stage=a.stage,exit_status=0,seconds=time.monotonic()-begun,
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    except Exception:
        if out.exists():write(out/'stage_error.json',dict(stage=a.stage,exit_status=1,error=traceback.format_exc()))
        raise
if __name__=='__main__':main()
