"""Static official-source/weight metadata compatibility audit; no foreign code executed."""
import argparse,hashlib,json
from pathlib import Path
from dunhuang.io import write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh

def run(source,out,replay=None):
    out=fresh(out);source=Path(source);m=read(source/"manifest.json")
    for repo in ("trgvy23/DanceFusion","setarehc/diffusion-motion-inbetweening"):
        folder=source/repo.split("/")[-1]
        for item in m[repo]["files"]:
            assert hashlib.sha256((folder/item["path"]).read_bytes()).hexdigest()==item["sha256"]
    d=m["trgvy23/DanceFusion"];tree=d["tree_paths"];model=(source/"DanceFusion/model/mdm.py").read_text()
    assert "4800 * 90" in model and "hframes" in model and "fut_frames" in model
    missing=[name for name in ("data_loaders/get_data.py","data_loaders/d2m/quaternion.py","data_loaders/d2m/normalizer.py","data_loaders/d2m/preprocess.py") if name not in tree]
    args=read(source/"pretrained_metadata/args.json")
    result=dict(dancefusion_commit=d["commit"],condmdi_commit=m["setarehc/diffusion-motion-inbetweening"]["commit"],
        source_files=d["files"],missing_required_modules=missing,music_shape="flattened 4800x90",motion_shape="B,151,1,T",
        edge_shape="B,150,151 plus B,150,4800 music",same_151_dimension_does_not_verify_coordinate_scaler_or_clock=True,
        published_checkpoint_available=m["pretrained_range_probe"]["status"]==206,checkpoint_archive_contents=m.get("pretrained_archive_listing"),
        official_pretrained_args=args,full_checkpoint_downloaded=False,
        runtime_compatible=False,normalizer_and_coordinate_parity_verified=False,official_inference_executed=False,
        status="NOT_RUN_OFFICIAL_DATA_MODULES_AND_NORMALIZATION_MISSING",optimizer_steps=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),source_manifest=ref(source/"manifest.json"),
        downloaded_sources_are_not_vendored_or_executed=True,normalizer_not_substituted_from_EDGE=True))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R13_OFFICIAL_AUDIT",result["status"],missing,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(a.source,a.out,a.replay_from)
