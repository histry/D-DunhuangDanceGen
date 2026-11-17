"""Offline, source-clock-bound paired-event viewing; no generated dance is implied."""
import argparse,base64,gzip,json,math,shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from dunhuang.io import checked,sha
from dunhuang.research.complete_r36.core import read,write,relative_fk


def run(data,out,alignment=None,all_sources=False):
    data=Path(data);out=Path(out)
    if out.exists():raise ValueError('Fresh viewing directory')
    out.mkdir(parents=True);(out/'audio').mkdir();manifest=read(data/'manifest.json');skeleton=np.load(data/'skeleton.npz');J=skeleton['J'];parents=skeleton['parents'];cfg=manifest['config'];rows={r['event_uid']:r for r in manifest['rows']};result=[]
    selected=manifest['sources'] if all_sources else [s for s in manifest['sources'] if s['source_uid'] in manifest['viewer_source_uids']]
    for s in selected:
        z=np.load(checked(s['source_arrays']));lo,hi=s['shared_source_interval'];P=relative_fk(z['R'][lo:hi],J,parents).astype('<f4');audio=checked(s['audio']);info=sf.info(audio);a=round((lo+s['audio_offset_frames'])/30*info.samplerate);b=round((hi+s['audio_offset_frames'])/30*info.samplerate)
        if a<0 or b>info.frames or b<=a:raise ValueError('Waveform source clock support')
        wave,sr=sf.read(audio,start=a,stop=b,dtype='float32',always_2d=True);wave=wave.mean(-1);g=math.gcd(sr,22050);wave=resample_poly(wave,22050//g,sr//g).astype(np.float32);name='audio/'+s['source_uid']+'.wav';sf.write(out/name,wave,22050,subtype='PCM_16')
        if abs(len(wave)/22050-(hi-lo)/30)>1/22050+1e-5:raise ValueError('Media duration does not match source frames')
        # Display-only waveform: every value is computed from the actual exported media.
        n=min(1200,len(wave));waveform=[float(np.max(abs(v))) for v in np.array_split(wave,n)]
        result.append(dict(source_uid=s['source_uid'],recording_uid=s['recording_uid'],role=s['role'],group=s['source_group'],source_start_frame=lo,frames=hi-lo,fps=30,audio=name,audio_offset_frames=s['audio_offset_frames'],original_audio_sha256=s['audio']['sha256'],native_sha256=s['native']['sha256'],source_clock=s['source_clock'],position_shape=list(P.shape),positions=base64.b64encode(P.tobytes()).decode(),waveform=waveform,events=[rows[e] for e in s['events']],phases=s['phases'],activity={k:z[k][lo:hi].tolist() for k in ('upper','lower','trunk','global','root_speed') if k in z},scope='REAL_CONTINUOUS_SOURCE_AND_KINEMATIC_CORES; NOT_GENERATED; COMMON22_ROOT_RELATIVE_BODY; NO_CONTACT_TRUTH'))
    payload=dict(schema='paired_event_r41_view_v1',parents=parents.tolist(),sources=result,all_events=manifest['rows'],summary={k:manifest[k] for k in ('role_counts','source_counts','music_groups','role_isolation','shared_choreography_across_roles','exclusions','event_label_scope','derivatives','test_tensor_reads','human_event_truth')},selection=manifest['viewer_selection'],alignment=read(alignment) if alignment else None)
    raw=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode();(out/'data.js').write_text('window.R41_DATA="'+base64.b64encode(gzip.compress(raw,mtime=0)).decode()+'";\n',encoding='utf8')
    for file in ('index.html','viewer.js','style.css'):shutil.copyfile(Path(__file__).parent/file,out/file)
    write(out/'protocol.json',dict(schema='paired_event_r41_view_v1',source_manifest_sha256=sha(data/'manifest.json'),sources=len(result),total_sources=len(manifest['sources']),total_events=len(manifest['rows']),phase_truth='KINEMATIC_PROXY_ONLY',generated_dance=False,all_sources_packed=all_sources,files=[dict(name=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
    print('R41_VIEWER',len(result),'sources',len(manifest['rows']),'events',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--alignment');p.add_argument('--all-sources',action='store_true');a=p.parse_args();run(a.data,a.out,a.alignment,a.all_sources)
