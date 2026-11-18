"""Offline synchronized music and saved ChangE reference positions."""
import base64
import html
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from dunhuang.research.complete_r36.core import relative_fk, rotation_sample, sha, write


def render(out, previews, audio_paths, library, joints, parents):
    out = out/"viewer"
    out.mkdir()
    data = []
    for slot, top, baseline in previews:
        a,b = slot["interval"]
        song = slot["song"]
        name = f"dunhuangwu{song+2}_preview.wav"
        with sf.SoundFile(audio_paths[song]) as source:
            rate = source.samplerate
            source.seek(round(a*rate/30))
            audio = source.read(round(b*rate/30)-round(a*rate/30))
        sf.write(out/name, audio, rate)
        panels = []
        for row in (top, baseline):
            warp = np.asarray(row["time_warp"])
            source = np.interp(np.arange(a,b),warp[:,1],warp[:,0])
            rotations = rotation_sample(library.rotations[row["source_uid"]], source)
            points = relative_fk(rotations, joints, parents).astype("<f4")
            panels.append(dict(source=row["source_uid"], event=row["event_uid"], warp=warp.tolist(),
                               score=row["score"], activity_score=row["heuristic"],
                               data=base64.b64encode(points.tobytes()).decode("ascii")))
        data.append(dict(audio=name, a=a, b=b, panels=panels,
                         candidates=[{k:r[k] for k in ("source_uid","event_uid","score","valid_frames")}
                                     for r in slot["candidates"][:8]]))
    (out/"data.js").write_text("window.R49="+json.dumps(dict(cases=data, parents=parents.tolist()))+";",encoding="utf8")
    page = Path(__file__).with_name("viewer.html").read_text(encoding="utf8")
    (out/"index.html").write_text(page,encoding="utf8")
    write(out/"protocol.json", dict(cases=len(data), fps=30, selected="fixed slot nearest song midpoint",
                                    scope="Retimed source preview", files=[
                                        dict(name=p.name, size=p.stat().st_size, sha256=sha(p))
                                        for p in sorted(out.iterdir())]))
