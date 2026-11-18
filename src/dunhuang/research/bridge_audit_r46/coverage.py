"""Whole-recording activity coverage, not a new event segmentation or selection policy."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,sha,Library
from dunhuang.research.reperformance_r44.phase import UPPER

def run(main,out):
    rows=read(Path(main)/'result.json')['records'];catalog=read('src/dunhuang/research/complete_r36/config.json')['catalog'];lib=Library(catalog);records=[]
    for r in rows:
        if r['source']['domain']!='ChangE':continue
        uid=r['source']['recording_uid'];R=lib.rotations[uid];v=Rotation.from_matrix((R[:-1].swapaxes(-1,-2)@R[1:]).reshape(-1,3,3)).magnitude().reshape(-1,22)[:,UPPER].mean(-1)*30;n=len(R);s=np.r_[0,np.cumsum(v)];window=(s[255:]-s[:-255])/255;start=int(np.argmax(window));current=int(r['source']['start']);interval=r['source']['source_interval'];records.append(dict(source_uid=uid,source_frames=n,catalog_source_sha256=lib.rows[uid][0]['source_sha256'],current_interval=interval,current_coverage_fraction=(interval[1]-interval[0])/n,current_mean_upper_speed_radps=float(window[current]),whole_recording_mean_upper_speed_radps=float(v.mean()),whole_recording_p95_upper_speed_radps=float(np.quantile(v,.95)),maximum_mean_activity_256frame_interval=[start,start+256],maximum_mean_activity_256frame_upper_speed_radps=float(window[start]),current_R45_detector=r['source']['audit']['capacity'],scope='FULL_NATIVE_SOURCE_ACTIVITY_DIAGNOSTIC_ONLY; MAX_ACTIVITY_WINDOW_NOT_SEMANTIC_EVENT_NOT_USED_TO_SELECT_MODEL_OR_TRAIN'))
    write(out,dict(records=records,catalog_sha256=sha(catalog),recording_frames=sum(r['source_frames'] for r in records),new_segmentation=False,model_selection=False,planner_routing_changed=False,optimizer_steps=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.main,a.out)
