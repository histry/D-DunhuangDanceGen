"""Post-hoc proxy event localization; not semantic truth or a new qualification gate."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.data.change_event_phases import motion_profiles,detect
from dunhuang.research.source_capacity.common import read,commit,fresh
from .reference import build

def partial(expected,actual,band=6):
    n,m=len(expected),len(actual);scores={(0,0):(0,0.)};paths={(0,0):[]}
    for i in range(n+1):
        for j in range(m+1):
            if (i,j) not in scores:continue
            actions=[]
            if i<n:actions.append((i+1,j,False))
            if j<m:actions.append((i,j+1,False))
            if i<n and j<m and expected[i]["type"]==actual[j]["type"] and abs(expected[i]["frame"]-actual[j]["frame"])<=band:actions.append((i+1,j+1,True))
            for a,b,match in actions:
                error=abs(expected[i]["frame"]-actual[j]["frame"]) if match else 0
                candidate=(scores[(i,j)][0]+int(match),scores[(i,j)][1]-error)
                if (a,b) not in scores or candidate>scores[(a,b)]:
                    scores[(a,b)]=candidate;paths[(a,b)]=paths[(i,j)]+([dict(expected_frame=expected[i]["frame"],generated_frame=actual[j]["frame"],type=expected[i]["type"],error_frames=error)] if match else [])
    return paths[(n,m)]

def run(single,out):
    out=fresh(out);p=read(Path(single)/"protocol.json");cfg=p["config"];phase_cfg=read("configs/data_event_v3.json")["phases"];records=[]
    for row,binding in zip(read(Path(single)/"result.json")["records"],p["bindings"]):
        native=np.load(checked(binding["native"]));plan=read(checked(binding["plan"]));_,_,_,expected=build(plan,cfg["guide"])
        expected=[dict(frame=e["music_frame"],type=e["type"]) for e in expected if e["type"]!="controlled_core_maximum" and 2<=e["music_frame"]<len(native)-2]
        expected=sorted({(e["frame"],e["type"]) for e in expected});expected=[dict(frame=t,type=k) for t,k in expected]
        detected=[e for e in detect(motion_profiles(native,list(range(22))),phase_cfg) if 2<=e["frame"]<len(native)-2]
        matches=partial(expected,detected)
        records.append(dict(pair=row["pair"],music=row["music"],arm=row["arm"],source_label=row["source_label"],
            expected=len(expected),generated=len(detected),matches=matches,proxy_recall=len(matches)/len(expected) if expected else None,
            proxy_precision=len(matches)/len(detected) if detected else None,proxy_timing_error_frames=float(np.mean([m["error_frames"] for m in matches])) if matches else None))
    write(out/"result.json",dict(records=records,scope="AUTOMATIC_KINEMATIC_PROXY_ONLY",match_band_frames=6,posthoc_diagnostic=True,used_for_weight_selection=False,used_for_qualification=False,new_generation_calls=0,human_reviewed=False))
    write(out/"protocol.json",dict(code_commit=commit(),single=ref(Path(single)/"protocol.json"),detector_config=phase_cfg))
    print("R8_PHASE_PROXY_DIAGNOSTIC",len(records),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--single",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(a.single,a.out)
