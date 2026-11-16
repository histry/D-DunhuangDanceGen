"""Compare scientific PCM/features/motion exactly while preserving both WAV container hashes."""
import argparse,copy,hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from .common import *

def pcm_record(record):
    path=checked(record);wave,sr=sf.read(path,dtype="float32",always_2d=True)
    return dict(file_name=Path(path).name,pcm_sha256=hashlib.sha256(wave.tobytes()+str(sr).encode()).hexdigest(),samples=len(wave),channels=wave.shape[1],sample_rate=sr),wave

def canonical_result(folder):
    result=copy.deepcopy(read(Path(folder)/"result.json"));containers=[]
    for row in result["records"]:
        source=row["music_window"];canonical,wave=pcm_record(source);containers.append(source);row["music_window"]=canonical
    return result,containers

def verify_generation(original,replay):
    a,ac=canonical_result(original);b,bc=canonical_result(replay);assert a==b
    for x,y in zip(ac,bc):
        px,wx=pcm_record(x);py,wy=pcm_record(y);assert px==py and np.array_equal(wx,wy)
    same_arrays(Path(original)/"arrays.npz",Path(replay)/"arrays.npz")
    return dict(all_non_container_fields_exact=True,audio_PCM_exact=True,music35_and_motion_arrays_exact=True,
        noise_and_model_and_time_bindings_exact=True,optimizer_steps=0,original_WAV_containers=ac,replay_WAV_containers=bc,
        explanation="libsndfile FLOAT WAV PEAK chunk stores creation timestamp. File-byte SHA remains audited per file; scientific comparison uses exact decoded PCM, feature35 and all generated arrays.")

def run(root,out):
    out=fresh(out);root=Path(root);forbid_optimizers();result=verify_generation(root/"plan_generation",root/"plan_generation_replay")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),original=ref(root/"plan_generation/result.json"),replay=ref(root/"plan_generation_replay/result.json")))
    print("R34_PLAN_REPLAY_ACCEPTED","PCM/features/noise/motion exact; both container hashes retained",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(a.root,a.out)
