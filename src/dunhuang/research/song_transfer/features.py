"""Whole-song official Layer66 extraction, frozen cache binding and endpoint replay."""
import argparse,sys,hashlib
import importlib.metadata
from pathlib import Path
import numpy as np
import soundfile as sf
from dunhuang.io import checked,pinned,sha,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.source_expression.review import audio_clip
def run(config,music,out,verify=False):
    inventory=read(checked(read(Path(music)/"protocol.json")["inventory"]));assets=read(config["assets"])
    root=pinned(assets["edge_root"],assets["edge_commit"]);sys.path.insert(0,str(root))
    from data.audio_extraction.jukebox_features import extract
    import jukemirlib
    from jukemirlib.constants import CACHE_DIR
    producer=dict(package_version=importlib.metadata.version("jukemirlib"),
        package_sources=[ref(p) for p in sorted(Path(jukemirlib.__file__).parent.glob("*.py"))],
        weights=[ref(Path(CACHE_DIR)/n) for n in ("vqvae.pth.tar","prior_level_2.pth.tar")],
        official_extractor=ref(root/"data/audio_extraction/jukebox_features.py"),publisher_provenance_verified=False)
    if verify:
        out=Path(out);receipt=read(out/"protocol.json");calls=0
        if receipt["producer_runtime"]!=producer:raise ValueError("Jukebox producer weights or runtime differ")
        for song in receipt["songs"]:
            values=np.load(checked(song["features"]))
            for record in song["slices"]:
                checked(record["audio"])
            if song["producer"]=="NEW_OFFICIAL_LAYER66":
                for index in sorted({0,len(values)-1}):
                    record=song["slices"][index];value,_=extract(str(checked(record["audio"])),skip_completed=False,dest_dir=str(out/"cache_verify"))
                    if not np.array_equal(np.asarray(value,np.float32),values[index]):raise ValueError("Independent endpoint Jukebox extraction differs")
                    calls+=1
        write(out/"replay.json",dict(status="FROZEN_WHOLE_AUDIO_FEATURE_PROVENANCE_AND_ENDPOINT_REPLAY",endpoint_extractions=calls,optimizer_calls=0));print("R7_JUKEBOX_VERIFIED",calls,flush=True);return
    out=fresh(out);songs=[]
    for song in inventory:
        original=checked(song["audio"]);clock=song["clock"];folder=out/str(song["case"]);folder.mkdir()
        frozen=read(checked(assets["feature_receipt"]));binding=frozen.get("original_audio",frozen.get("audio"))
        if binding and binding["sha256"]==song["audio"]["sha256"] and frozen["clock"]==clock and frozen["layer"]==66:
            file=checked(assets["features"]);value=np.load(file)
            if value.shape!=(clock["windows"],150,4800) or not np.isfinite(value).all():raise ValueError("Frozen feature clock differs")
            songs.append(dict(case=song["case"],audio=song["audio"],features=ref(file),feature_receipt=assets["feature_receipt"],clock=clock,producer="HASH_VERIFIED_FROZEN_LAYER66",slices=[]));continue
        x,sr=sf.read(original,dtype="float32",always_2d=True);mono=x.mean(1);padded=np.pad(mono,(0,clock["silence_samples"]));values=[];slices=[]
        for i in range(clock["windows"]):
            a=round(i*2.5*sr);data=padded[a:a+5*sr,None];file=folder/("slice_%03d.wav"%i);audio_clip(file,data,sr)
            v,_=extract(str(file),skip_completed=False,dest_dir=str(folder/"cache"));v=np.asarray(v,np.float32)
            if v.shape!=(150,4800) or not np.isfinite(v).all() or not v.any():raise ValueError("Real full-song Jukebox condition required")
            values.append(v);slices.append(dict(audio=ref(file),source_start_sample=a,observed_end_sample=min(len(mono),a+5*sr),silence_samples=max(0,a+5*sr-len(mono)),features_sha256=hashlib.sha256(v.tobytes()).hexdigest()))
            print("R7_JUKEBOX_WINDOW",song["case"],i+1,clock["windows"],flush=True)
        file=folder/"features.npy";np.save(file,np.stack(values))
        songs.append(dict(case=song["case"],audio=song["audio"],features=ref(file),clock=clock,producer="NEW_OFFICIAL_LAYER66",slices=slices))
    write(out/"protocol.json",dict(code_commit=commit(),config=config,music_protocol=ref(Path(music)/"protocol.json"),producer_runtime=producer,layer=66,fps=30,songs=songs,scope=config["scope"]))
    print("R7_WHOLE_JUKEBOX_COMPLETED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--music",required=True);p.add_argument("--out",required=True);p.add_argument("--verify",action="store_true");a=p.parse_args();run(read(a.config),a.music,a.out,a.verify)
