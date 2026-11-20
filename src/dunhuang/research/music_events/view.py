"""Full-song listening and model-event overlay; labels are explicitly unvalidated."""
from pathlib import Path
import json,shutil
import numpy as np
from .infer import ROOT
from dunhuang.research.local_timing_r68.common import read,write,sha

def build():
    source=ROOT/'main';out=ROOT/'viewer';out.mkdir(exist_ok=False);(out/'audio').mkdir();tracks=[];samples=[]
    for directory in sorted(p for p in source.iterdir() if p.is_dir() and (p/'catalog.json').exists()):
        c=read(directory/'catalog.json');raw=Path(c['audio_path']);assert sha(raw)==c['audio_sha256'];dest=out/'audio'/raw.name;shutil.copyfile(raw,dest)
        with np.load(directory/'activations.npz') as z:curves={k:z[k].tolist() for k in ['acoustic_energy','acoustic_onset','acoustic_accent','acoustic_chroma_novelty']}
        tracks.append(dict(name=raw.stem,audio='audio/'+raw.name,sha=c['audio_sha256'],duration=c['duration_seconds'],events=c['events'],comparison=c['comparison'],segments=c['segments'],curves=curves,phrase_status=c['phrase_status']))
        # Fixed equal-position samples plus deterministic disagreement excerpts, not quality-selected.
        disagreement=c['comparison']['beat_this_vs_allin1_beat'];bt=[e['seconds'] for e in c['events'] if e['source']=='beat_this' and e['kind']=='beat'];times=[c['duration_seconds']*q for q in [.25,.5,.75]]+[bt[i] for i in disagreement['unmatched_a'][:3]]
        samples.extend(dict(track=raw.stem,start=max(0,t-4),end=min(c['duration_seconds'],t+4),reason='UNIFORM_POSITION' if i<3 else 'MODEL_DISAGREEMENT',human_checked=False) for i,t in enumerate(times))
    payload=dict(tracks=tracks,source_manifest_sha256=sha(source/'manifest.json'),human_records=0,can_route=False)
    template=Path(__file__).with_name('viewer.html').read_text(encoding='utf8')
    (out/'index.html').write_bytes(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':'))).encode())
    write(out/'review_samples.json',samples)
    write(out/'protocol.json',dict(source_manifest_sha256=payload['source_manifest_sha256'],full_original_audio=True,human_records=0,model_agreement_not_accuracy=True,phrase_truth='UNKNOWN',can_route=False))
    write(ROOT/'viewer_delivery.json',dict(files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()]))
    print('MUSIC_VIEW_READY',len(tracks),flush=True)

if __name__=='__main__':build()
