"""Offline source-recording and exact half-open ChangE cut inspection, no generation."""
import argparse,base64,gzip,hashlib,json,shutil
from pathlib import Path
import numpy as np
from dunhuang.contracts import matrices

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fk(R,J,parents):
    p=[np.zeros((len(R),3))];world=[R[:,0]]
    for j in range(1,22):
        k=int(parents[j]);p.append(p[k]+(world[k]@(J[j]-J[k])[None,:,None])[...,0]);world.append(world[k]@R[:,j])
    return np.stack(p,1).astype("<f4")

def build(catalog,sidecar,rig,out):
    catalog=Path(catalog);sidecar=Path(sidecar);rig=Path(rig);out=Path(out)
    if out.exists():raise ValueError("Fresh output required")
    old=json.loads(catalog.read_text(encoding="utf8"));new=json.loads(sidecar.read_text(encoding="utf8"));sk=json.loads(rig.read_text(encoding="utf8"))
    assert sha(sidecar)==sk["sidecar_sha256"]
    by={e["event_uid"]:e for e in old["entries"]};assert len(by)==len(new["entries"])==499
    J=np.asarray(sk["J22"]);parents=sk["parents"];assert J.shape==(22,3) and len(parents)==22
    sources=[];events=[];checks=[]
    uids=list(dict.fromkeys(e["source_uid"] for e in old["entries"]))
    for uid in uids:
        rows=sorted((e for e in new["entries"] if e["source_uid"]==uid),key=lambda e:(e["source_interval"][0],e["event_uid"]))
        legacy=by[rows[0]["event_uid"]];path=catalog.parent/Path(legacy["reference"]["path"]).name
        assert sha(path)==rows[0]["reference"]["sha256"]==legacy["reference"]["sha256"]
        native=np.load(path,allow_pickle=False);assert native.shape[1]==151 and np.isfinite(native).all()
        R=matrices(native[:,7:].reshape(-1,24,6))[:,:22]
        points=fk(R,J,parents);assert np.max(abs(points[:,0]))==0
        # Anatomical lengths must remain fixed throughout this display conversion.
        err=max(float(abs(np.linalg.norm(points[:,j]-points[:,parents[j]],axis=-1)-np.linalg.norm(J[j]-J[parents[j]])).max()) for j in range(1,22))
        assert err<1e-5
        ids=[];coverage=np.zeros(len(points),np.int16)
        for e in rows:
            original=by[e["event_uid"]];a,b=e["source_interval"]
            assert original["source_interval"]==[a,b] and original["source_sha256"]==e["source_sha256"] and original["reference"]["sha256"]==e["reference"]["sha256"]
            assert 0<=a<b<=len(points) and b-a==e["duration_frames"]
            ids.append(e["event_uid"]);coverage[a:b]+=1
            phases=[p for p in e["kinematic_phases"] if a<=p["frame"]<b]
            events.append(dict(event_uid=e["event_uid"],source_uid=uid,source_interval=[a,b],duration_frames=b-a,
                category=original.get("semantic",{}).get("category",uid),safe_anchor=e["safe_anchor"],anchor_status=e["anchor_status"],
                original_peak=e.get("original_peak"),controlled_peak=e.get("controlled_peak"),phases=phases,
                source_sha256=e["source_sha256"],native_sha256=e["reference"]["sha256"],
                slice_policy="EXACT_SOURCE_FRAMES_[START,END); ORIGINAL_30FPS; NO_E1W; NO_GENERATION"))
            # A cut is a view into this one source track, not a second lossy conversion.
            assert np.array_equal(points[a:b][0],points[a]) and np.array_equal(points[a:b][-1],points[b-1])
        sources.append(dict(uid=uid,category=events[-len(rows)]["category"],frames=len(points),fps=30,event_uids=ids,
            track=dict(shape=list(points.shape),data=base64.b64encode(points.tobytes()).decode()),native_sha256=sha(path),source_sha256=rows[0]["source_sha256"],
            uncovered_frames=int((coverage==0).sum()),overlap_frames=int((coverage>1).sum()),
            range_x=np.ptp(points[:,[16,17,18,19,20,21]],axis=0).tolist()))
        checks.append(dict(source_uid=uid,frames=len(points),events=len(rows),native_sha256=sha(path),bone_length_max_error_m=err,exact_source_slice=True))
    payload=dict(schema="change_source_and_cuts_v1",fps=30,parents=parents,sources=sources,events=events,Root_truth="UNKNOWN",
        display="ORIGINAL_BODY_ROTATIONS_ON_SHARED22_RIG_ROOT_RELATIVE; NO_SMOOTHING",audio="UNPAIRED_NO_INVENTED_MUSIC",generation=False)
    encoded=json.dumps(payload,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
    out.mkdir(parents=True);(out/"data.js").write_text('window.CATALOG_B64="'+base64.b64encode(gzip.compress(encoded,mtime=0)).decode()+'";\n',encoding="utf8")
    templates=Path(__file__).resolve().parents[1]/"src/dunhuang/visualization/change_catalog"
    for n in ("index.html","viewer.js","style.css"):shutil.copyfile(templates/n,out/n)
    receipt=dict(source_catalog_sha256=sha(catalog),phase_sidecar_sha256=sha(sidecar),rig_sha256=sha(rig),
        source_rig_provenance=sk,event_count=len(events),source_count=len(sources),source_checks=checks,
        all_499_intervals_exact=True,no_generated_motion=True,no_resampling=True,no_smoothing=True,
        Root_truth="UNKNOWN",review="DISPLAY_ONLY_NOT_HUMAN_LABELS",files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir()})
    (out/"protocol.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(dict(sources=len(sources),events=len(events),total_frames=sum(x["frames"] for x in sources),viewer=str(out/"index.html"))))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--catalog",required=True);p.add_argument("--sidecar",required=True);p.add_argument("--rig",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    build(a.catalog,a.sidecar,a.rig,a.out)
