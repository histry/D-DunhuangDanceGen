"""Saved short choreography on its actual music timeline."""
import base64
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from dunhuang.research.complete_r36.core import read,write,sha


def render(root,songs):
    out = root/"viewer"
    out.mkdir()
    result = read(root/"result.json")
    arrays = np.load(root/"arrays.npz",allow_pickle=False)
    cases = []
    for song in range(3):
        for count in (2,3):
            bank = [r for r in result["cases"] if r["song"]==song and r["events"]==count
                    and r["status"]=="NUMERICAL_SHORT_PLAN"]
            if not bank:
                continue
            r = min(bank,key=lambda r:(abs(r["transition_frames"]-48),r["transition_frames"]))
            name = r["key"]+".wav"
            a,b = r["music_interval"]
            with sf.SoundFile(songs[song]["path"]) as f:
                sr = f.samplerate
                f.seek(round(a*sr/30))
                sound = f.read(round(b*sr/30)-round(a*sr/30))
            sf.write(out/name,sound,sr)
            cases.append(dict(**r,audio=name,data=base64.b64encode(arrays[r["key"]].tobytes()).decode()))
    (out/"data.js").write_text("window.R50="+json.dumps(dict(cases=cases,parents=arrays["parents"].tolist()))+";",encoding="utf8")
    (out/"index.html").write_text(Path(__file__).with_name("viewer.html").read_text(encoding="utf8"),encoding="utf8")
    write(out/"protocol.json",dict(cases=len(cases),files=[
        dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())]))
