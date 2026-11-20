"""R69 named development playback and a separately projected blind review pack.

The private seed and method map never enter either reviewer HTML or its media
manifest. Building pages creates no human ratings and authenticates no person.
"""
from collections.abc import Mapping
import hashlib
import hmac
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from dunhuang.research.complete_r36.core import relative_fk
from dunhuang.research.warp_coupling_r53.common import array_hash

METHODS = ("Uniform", "E1W", "BeatDP", "RhythmDTW", "Single64", "Multi64")
LABELS = {"Uniform":"匀速", "E1W":"冻结 E1-W", "BeatDP":"拍点 DP", "RhythmDTW":"节奏 DTW",
          "Single64":"冻结单尺度64读出", "Multi64":"冻结多尺度64读出"}
DIMENSIONS = ("music_coordination", "transition_timing", "continuity", "richness", "naturalness")


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def _write(path,value):
    Path(path).write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode("utf8"))


def _opaque(seed,kind,value):
    body=json.dumps([kind,value],ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
    return hmac.new(seed,body.encode("utf8"),hashlib.sha256).hexdigest()


def _rotations(value,n=None):
    x=np.asarray(value,dtype=float)
    if (x.ndim!=4 or x.shape[1:]!=(22,3,3) or len(x)<2 or n is not None and len(x)!=n
            or not np.isfinite(x).all()
            or not np.allclose(x.swapaxes(-1,-2)@x,np.eye(3),atol=1e-5,rtol=0)
            or not np.allclose(np.linalg.det(x),1,atol=1e-5,rtol=0)):
        raise ValueError("Actual proper common22 rotations with registered length required")
    return x


def _position(value,J,parents,n):
    x=_rotations(value,n);p=relative_fk(x,J,parents)
    if p.shape!=(n,22,3) or not np.isfinite(p).all():
        raise ValueError("Actual finite body22 FK required")
    return np.round(p,5).tolist()


def _clock(case,clocks):
    own=[clocks.get(str(case.get("songidx"))),clocks.get(case.get("music_id"))]
    own.extend(clocks.values())
    for clock in own:
        if (isinstance(clock,Mapping) and clock.get("audio_path")
                and clock.get("audio_sha256")==case.get("audio_sha256")):
            return clock
    raise ValueError("Actual identified full audio clock required")


def _scalar(value):
    return value is None or isinstance(value,(str,bool,int,float))


def _summary(row):
    result={k:v for k,v in row.items() if _scalar(v)}
    for key in ("connection","costs","metrics"):
        value=row.get(key)
        if isinstance(value,Mapping):
            result[key]={k:v for k,v in value.items() if _scalar(v)}
    result.update(artistic_quality="NOT_EVALUABLE",Native24="NOT_EVALUABLE",world_root_contact="UNKNOWN")
    return result


def _audio(watch,case,clock,interval,known,raw_hashes):
    A,B=map(int,interval)
    if B-A!=528 or not 0<=A<B<=clock["frames"]:
        raise ValueError("Actual full528 pair audio support required")
    identity=case["audio_sha256"]
    if not isinstance(identity,str) or not re.fullmatch(r"[a-f0-9]{64}",identity):
        raise ValueError("Actual raw audio SHA256 required")
    name=identity[:12]+"_"+str(A)+"_"+str(B)+".wav"
    if name not in known:
        raw=Path(clock["audio_path"])
        if raw not in raw_hashes:raw_hashes[raw]=_sha(raw)
        if raw_hashes[raw]!=identity:raise ValueError("Identified full audio bytes changed")
        subprocess.run(["ffmpeg","-nostdin","-v","error","-i",str(raw),"-ss",str(A/30),
                        "-t",str(528/30),"-vn","-map_metadata","-1","-c:a","pcm_s16le",
                        "-ac","1","-ar","22050",str(watch/name)],check=True)
        known[name]=dict(path="audio/"+name,bytes=(watch/name).stat().st_size,
                         sha256=_sha(watch/name),raw_audio_sha256=identity,pair_music_interval=[A,B])
    return known[name]


def _pair_interval(case,panels):
    pair=case.get("pair_music_interval")
    candidates=[(p.get("row",{}).get("connection") or {}).get("pair_music_interval") for p in panels]
    candidates=[p for p in candidates if p is not None]
    if pair is None and candidates:pair=candidates[0]
    if (not isinstance(pair,(list,tuple)) or len(pair)!=2
            or any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) for v in pair)
            or pair[1]-pair[0]!=528 or pair[0]<0 or any(list(p)!=list(pair) for p in candidates)):
        raise ValueError("One exact parent528 audio interval required")
    return list(map(int,pair))


def _review_projection(payload):
    """Reject metadata leakage even if a future caller adds named fields."""
    allowed={"pack_id","protocol_sha256","fps","frames","parents","dimensions","form_id","cases"}
    if set(payload)!=allowed:raise ValueError("Anonymous top-level whitelist violated")
    for item in payload["cases"]:
        if set(item)!={"blind_item_id","audio","variants"}:raise ValueError("Anonymous case whitelist violated")
        if set(item["audio"])!={"path","media_sha256"}:raise ValueError("Anonymous audio whitelist violated")
        for variant in item["variants"]:
            if set(variant)!={"blind_variant_id","available","P"}:raise ValueError("Anonymous variant whitelist violated")
    return payload


def _html(path,template,payload,mode):
    text=template.read_text(encoding="utf8")
    if text.count("__DATA__")!=1:raise ValueError("Exactly one viewer payload slot required")
    encoded=json.dumps(payload,ensure_ascii=False,separators=(",",":"),allow_nan=False)
    encoded=encoded.replace("<","\\u003c").replace("\u2028","\\u2028").replace("\u2029","\\u2029")
    path.write_bytes(text.replace("__DATA__",encoded).replace("__MODE__",mode).encode("utf8"))


def build(out,display,J,parents,clocks,*,blind_seed):
    """display: (case, panels, native); panels have optional actual full_R528.

    The caller stores its private32-byte seed and supplies that same seed for
    independent replay. The seed is not generated from a public config or ID.
    """
    if not isinstance(blind_seed,bytes) or len(blind_seed)!=32:
        raise ValueError("A private32-byte seed is required; public experiment seeds are not accepted")
    out=Path(out);named=out/"viewer";blind=out/"reviewer";admin=out/"admin"
    for folder in (named,blind,admin):folder.mkdir(exist_ok=True)
    (named/"audio").mkdir();(blind/"audio").mkdir()
    J=np.asarray(J,float);parents=np.asarray(parents)
    if (J.shape!=(22,3) or not np.isfinite(J).all() or parents.shape!=(22,)
            or parents.dtype.kind not in "iu" or parents[0]!=-1
            or any(not 0<=parents[j]<j for j in range(1,22))):
        raise ValueError("Shared actual body22 skeleton required")
    rows=list(display)
    if not rows:raise ValueError("Registered nonempty display population required")
    pack_id="r69_"+_opaque(blind_seed,"pack","fixed-six-methods")[:24]
    protocol=dict(schema="R69_QUALITY_PROTOCOL_V1",pack_id=pack_id,form_id="F01",frames=528,fps=30,
        dimensions=list(DIMENSIONS),rating_values=[1,2,3,4,5],abstention="UNSURE_SEPARATE_FROM_UNRATED",
        cases=len(rows),variants_per_case=6,ordering="SEALED_OPAQUE_IDS_NO_RUNTIME_RESHUFFLE",
        camera="SHARED_FIXED_ROOT_RELATIVE_BODY22_190PX_PER_METRE_NO_PER_VARIANT_RESCALE",
        scope="ANONYMOUS_PRESENTATION_NOT_AUTHENTICATED_INDEPENDENT_REVIEW",
        evidence_status="PREPARATION_ONLY_NO_REVIEWS_CREATED",human_reviewed=False,
        artistic_quality="NOT_EVALUABLE",Native24="NOT_EVALUABLE",can_route=False)
    _write(blind/"protocol.json",protocol);protocol_sha=_sha(blind/"protocol.json")
    named_cases=[];blind_cases=[];map_cases=[];audio_files={};raw_hashes={};seen=set()
    for case,raw_panels,native in rows:
        if case["key"] in seen:raise ValueError("Repeated registered case identity")
        seen.add(case["key"])
        panel_list=list(raw_panels)
        if len(panel_list)!=6 or {p["method"] for p in panel_list}!=set(METHODS):
            raise ValueError("All six fixed methods and failure placeholders required")
        if len({p["method"] for p in panel_list})!=6:raise ValueError("Duplicate method result")
        pair=_pair_interval(case,panel_list);side=case.get("parent_side")
        if side not in ("left","right"):raise ValueError("Actual parent side required")
        core_start=0 if side=="left" else 288
        if case["target_interval"]!=[pair[0]+core_start,pair[0]+core_start+240]:
            raise ValueError("Selected Core and actual full-pair music clock differ")
        clock=_clock(case,clocks);audio=_audio(named/"audio",case,clock,pair,audio_files,raw_hashes)
        blind_audio_name=_opaque(blind_seed,"audio",[case["audio_sha256"],pair])[:32]+".wav"
        dest=blind/"audio"/blind_audio_name
        if not dest.exists():dest.write_bytes((named/audio["path"]).read_bytes())
        if _sha(dest)!=audio["sha256"]:raise ValueError("Anonymous audio differs from actual pair")
        by_method={p["method"]:p for p in panel_list};dev_panels=[];anon_variants=[];variants=[]
        bid="item_"+_opaque(blind_seed,"item",case["key"])[:24]
        order=sorted(METHODS,key=lambda m:_opaque(blind_seed,"order",[case["key"],m]))
        for method in METHODS:
            result=by_method[method];row=result.get("row") or {};R=result.get("R");full=result.get("full_R")
            raw_core=None if R is None else np.asarray(R)
            raw_full=None if full is None else np.asarray(full)
            q=result.get("q");a,b=case["source_interval"]
            if R is not None:
                R=_rotations(R,240);q=np.asarray(q,float)
                if q.shape!=(240,) or not np.isfinite(q).all() or not (np.diff(q)>0).all() or q[0]!=a or q[-1]!=b-1:
                    raise ValueError("Actual monotone selected-Core source clock required")
            elif q is not None:raise ValueError("No source clock may be shown without actual Core")
            if full is not None:
                full=_rotations(full,528)
                if (R is None or array_hash(raw_full[core_start:core_start+240])!=array_hash(raw_core)
                        or not np.array_equal(full[core_start:core_start+240],R)):
                    raise ValueError("Full pair must preserve the actual selected Core bytes")
            dev_panels.append(dict(method=method,label=LABELS[method],
                P=_position(full,J,parents,528) if full is not None else None,
                core_P=_position(R,J,parents,240) if R is not None else None,
                q=None if q is None else np.round(q,6).tolist(),core_start=core_start,
                status=row.get("status","UNKNOWN"),reason=row.get("reason"),diagnostics=_summary(row),
                full_pair_available=full is not None,core_only_available=R is not None))
        lookup={p["method"]:p for p in dev_panels}
        for method in order:
            result=by_method[method];p=lookup[method]
            vid="variant_"+_opaque(blind_seed,"variant",[case["key"],method])[:24]
            anon_variants.append(dict(blind_variant_id=vid,available=p["P"] is not None,P=p["P"]))
            variants.append(dict(blind_variant_id=vid,method=method,available=p["P"] is not None,
                                 trajectory_sha256=array_hash(np.asarray(result["full_R"])) if result.get("full_R") is not None else None))
        if not isinstance(native,Mapping) or native.get("fps")!=30:
            raise ValueError("Unretimed native source30FPS input required")
        n=case["source_interval"][1]-case["source_interval"][0]
        raw=_rotations(native["R"],n)
        named_cases.append(dict(key=case["key"],source_uid=case["source_uid"],source_interval=case["source_interval"],
            target_interval=case["target_interval"],pair_music_interval=pair,parent_side=side,core_start=core_start,
            audio=audio["path"],audio_sha256=audio["sha256"],source_audio_sha256=case["audio_sha256"],
            music=case.get("music_id",case.get("songidx")),panels=dev_panels,
            native=dict(P=_position(raw,J,parents,n),frames=n,fps=30,source_interval=case["source_interval"],retimed=False)))
        blind_cases.append(dict(blind_item_id=bid,audio=dict(path="audio/"+blind_audio_name,
            media_sha256=audio["sha256"]),variants=anon_variants))
        map_cases.append(dict(blind_item_id=bid,case_key=case["key"],source_uid=case["source_uid"],
            source_interval=case["source_interval"],pair_music_interval=pair,variants=variants))
    blind_cases.sort(key=lambda c:_opaque(blind_seed,"case_order",c["blind_item_id"]))
    map_cases.sort(key=lambda c:next(i for i,b in enumerate(blind_cases) if b["blind_item_id"]==c["blind_item_id"]))
    public=_review_projection(dict(pack_id=pack_id,protocol_sha256=protocol_sha,fps=30,frames=528,
        parents=parents.tolist(),dimensions=list(DIMENSIONS),form_id="F01",cases=blind_cases))
    dev=dict(frames=528,fps=30,parents=parents.tolist(),methods=list(METHODS),labels=LABELS,cases=named_cases)
    template=Path(__file__).with_name("viewer.html")
    _html(named/"index.html",template,dev,"named")
    # A separate template prevents hidden developer labels, data or controls
    # from leaking into the anonymous page through a generic toggle.
    reviewer_template=Path(__file__).with_name("reviewer.html")
    _html(blind/"index.html",reviewer_template,public,"anonymous")
    _write(admin/"map.json",dict(schema="R69_ADMIN_V1",pack_id=pack_id,protocol_sha256=protocol_sha,
        form_id="F01",seed_sha256=hashlib.sha256(blind_seed).hexdigest(),cases=map_cases,
        anonymous_case_order=[c["blind_item_id"] for c in blind_cases],human_reviewed=False,
        scope="PRIVATE_MAPPING_NOT_FOR_REVIEWERS_NOT_PART_OF_MEDIA_MANIFEST"))
    _write(named/"protocol.json",dict(schema="R69_NAMED_DEVELOPMENT_VIEWER_V1",methods=list(METHODS),
        frames=528,fps=30,core_mode_frames=240,core_start_by_side=dict(left=0,right=288),
        transition_frames=[240,288],source_speed_view="INDEPENDENT_SILENT30FPS_UNRETIMED",
        masks="CORE_AND_BRIDGE_REGIONS_SHOWN_ONLY_IN_NAMED_DEVELOPMENT_VIEW",
        artistic_quality="NOT_EVALUABLE",Native24="NOT_EVALUABLE",human_reviewed=False,can_route=False,
        audio_files=audio_files))
    files=[]
    for folder in (named,blind):
        for p in sorted(folder.rglob("*")):
            if p.is_file():files.append(dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=_sha(p)))
    _write(out/"viewer_manifest.json",dict(files=files,private_admin_included=False,
        raw_rotations=False,full_audio=False,weights=False,trajectories_in_html=True,
        frames=528,fps=30,quality_winner=None,human_reviewed=False,can_route=False))
    return named/"index.html"
