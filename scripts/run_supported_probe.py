"""Explicit narrower follow-up: original wrist failures remain in experiment_v1.
No pose clipping or normalizer refit. Joints 12..19 only, excluding both wrists.
"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.pipeline import run
from dunhuang.evaluation import evaluate
from dunhuang.io import write
root=Path(__file__).resolve().parents[1]
original=root/'outputs/experiment_v1'
out=root/'outputs/supported_upper8';out.mkdir(parents=True,exist_ok=False)
results=[]
for label in ['5s','15s','whole']:
    q=json.loads((original/('anchored_'+label)/'request.json').read_text())
    for anchor in q['anchors']:anchor['joints']=list(range(12,20))
    q['output']=str(out/label)
    print('START_UPPER8',label,flush=True)
    path=run(q)
    baseline=original/('baseline_'+label)
    write(path/'baseline_matched_evaluation.json',evaluate(np.load(baseline/'native.npy'),np.load(baseline/'native_fk.npy'),q['anchors'],q['beat_frames']))
    results.append(dict(case=label,**json.loads((path/'status.json').read_text())))
    write(out/'summary.json',dict(cases=results,scope='joints12..19; both wrists excluded after original preflight failure',
        original_failed_cases_retained=True,original_ten_joint_goal_met=False,training=False,full_generation_qualified=False))
    print('FINISH_UPPER8',label,results[-1],flush=True)
