"""Independent source-packet and Root-translation checks; no motion repair claims."""
import argparse,json,re,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import sha,write,checked
def read(p):return json.loads(Path(p).read_text())
def check(study,source):
    study=Path(study);source=Path(source);main=study/"main";protocol=read(main/"protocol.json")
    bindings=read(checked(protocol["bindings"]));old={i["blind_item_id"]:i for i in read(source/"items.json")};error=0.;frames=0;audio=0
    for binding in bindings:
        bid=binding["blind_item_id"];old_file=source/(bid+".html");new_file=main/"reviewer"/(bid+".html")
        assert sha(old_file)==old[bid]["media_sha256"] and sha(new_file)==binding["media_sha256"]
        previous=np.asarray(json.loads(re.search(r"const xyz=(.*?),parents=",old_file.read_text()).group(1)),float)
        current=json.loads(re.search(r"const motion=(.*?),parents=",new_file.read_text()).group(1))
        pose=np.asarray(current["pose"],float);world=np.asarray(current["world"],float)
        assert np.array_equal(pose,previous)
        native=np.load(checked(binding["native"]));lo,hi=binding["display_interval"]
        root=native[lo:hi,4:7].astype(float);offset=root-root[:1]
        expected=pose+offset[:,None,:];err=float(abs(world-expected).max());assert err<=1.01e-6;error=max(error,err);frames+=len(pose)
        if binding.get("audio"):
            assert sha(checked(binding["audio"]))==sha(source/(bid+".wav"));audio+=1
        assert binding["fitted_root_observation"]=="UNVERIFIED_FIT_NOT_WORLD_OR_CONTACT_TRUTH"
    for p in (main/"reviewer").iterdir():assert sha(p)==sha(study/"replay/reviewer"/p.name)
    write(study/"checks.json",dict(code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),verifier_sha256=sha(__file__),items=len(bindings),pose_frames_exact=frames,
        audio_assets_exact=audio,world_root_reconstruction_max_m=error,source_manifest_sha256=sha(source/"items.json"),human_reviewed=False,operational_qualified=False,
        smoothing=False,optimizer_calls=0,edge_forward_calls=0))
    print("VIEWER_V2_INDEPENDENT_CHECK",len(bindings),frames,error)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--study",required=True);p.add_argument("--source",required=True);a=p.parse_args();check(a.study,a.source)
