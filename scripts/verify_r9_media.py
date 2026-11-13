import argparse,subprocess,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read
def run(root):
    root=Path(root);p=read(root/"analysis/protocol.json");videos=[]
    for v in p["videos"]:
        f=checked(v["video"]);x=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-show_streams","-of","json",str(f)],text=True))
        stream=next(s for s in x["streams"] if s["codec_type"]=="video")
        assert int(stream["nb_read_frames"])==v["frames"] and stream["r_frame_rate"]=="30/1"
        videos.append(dict(name=v["name"],frames=v["frames"],sha256=v["video"]["sha256"]))
    assert len(videos)==48
    assert read(root/"analysis/result.json")==read(root/"analysis_replay/result.json")
    write(root/"analysis/media_checks.json",dict(videos=videos,count=len(videos),post_analysis_exact_replay=True,new_generation_calls=0))
    print("R9_MEDIA_VERIFIED",len(videos),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--main",required=True);a=p.parse_args();run(a.main)
