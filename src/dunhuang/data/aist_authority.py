"""Publisher-record identities and signal-based AIST audio clock calibration."""
import json,hashlib,re,zipfile,subprocess,csv
from pathlib import Path
from html.parser import HTMLParser
from xml.etree import ElementTree as ET
import numpy as np
from scipy.signal import correlate,resample_poly
import soundfile as sf
from dunhuang.io import checked,sha
from .sources import read,ref

NAME=re.compile(r'^(g[A-Z]{2})_(sBM|sFM)_cAll_(d[0-9]{2})_(m[A-Z]{2}[0-5])_(ch[0-9]{2})$')
POLICY=dict(schema='refined_audio_calibration_v1',rate=8000,anchor_seconds=3.,
    minimum_ncc=.98,maximum_residual_seconds=1/60,maximum_lag_spread_seconds=1/60,
    local_search_seconds=.1,maximum_clock_duration_error_seconds=.034,
    definition='source_audio_time = exact60fps_motion_time + calibrated_offset',
    ambiguity='periodic alternatives reported; publisher refined-origin plus measured prefix resolves zero only')

def choreography_table(path):
    # Tiny official XLSX table, read as XML without executing workbook content.
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        strings=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
        rows=[]
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//m:row',ns):
            cells=[]
            for cell in row.findall('m:c',ns):
                value=cell.find('m:v',ns)
                cells.append(strings[int(value.text)] if cell.get('t')=='s' else value.text if value is not None else '')
            rows.append(cells)
    result={}
    for genre,ch,title in rows[1:]:
        key=(genre.strip(),ch.strip())
        if key in result:raise ValueError('Duplicate official choreography key')
        result[key]=title.strip()
    if len(result)!=100:raise ValueError('Expected publisher 10 x 10 basic choreography table')
    return result

class VideoLinks(HTMLParser):
    def __init__(self):super().__init__();self.urls=set()
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            u=dict(attrs).get('href','')
            if u.startswith('https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/') and u.endswith('.mp4'):self.urls.add(u)

def authority(root):
    root=Path(root);record=read(root/'authority.json')
    for r in record['files']:checked(dict(path=str(root/r['file']),sha256=r['sha256']))
    parser=VideoLinks();parser.feed((root/'basic_links.html').read_text(encoding='utf8'))
    mapping=dict(line.split() for line in (root/'camera_mapping.txt').read_text().splitlines() if line.strip())
    metadata={}
    if (root/'refined_2M_sBM_all.csv').exists():
        with (root/'refined_2M_sBM_all.csv').open(encoding='utf8',newline='') as f:
            for row in csv.DictReader(f):
                metadata[row['URL']]=row;parser.urls.add(row['URL'])
    return dict(root=root,metadata=metadata,identity=sha(root/'authority.json'),table=choreography_table(root/'choreo.xlsx'),
        urls=parser.urls,mapping=mapping,all=set((root/'all.txt').read_text().split()))

def identify(row,auth):
    name=row['sequence_id'];m=NAME.fullmatch(name)
    if not m or name not in auth['all'] or name not in auth['mapping']:raise ValueError('Unregistered sequence')
    genre,situation,dancer,music,ch=m.groups()
    if music[1:3]!=genre[1:]:raise ValueError('Music/genre mismatch')
    if situation!='sBM':return dict(status='EXCLUDED_UNRESOLVED_ADVANCED_CHOREOGRAPHY',sequence_id=name)
    if (genre,ch) not in auth['table']:raise ValueError('Basic choreography not in publisher workbook')
    video=name.replace('cAll','c01')
    url='https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/'+video+'.mp4'
    compact=url.replace('/video/10M/','/video/2M/')
    if compact in auth['urls']:url=compact
    if url not in auth['urls']:raise ValueError('No official refined video link')
    meta=auth.get('metadata',{}).get(url)
    if meta:
        expected=dict(GENRE=genre,SITUATION=situation,CAMERA='c01',DANCER=dancer,MUSIC=music,CHOREOGRAPHY=ch)
        if any(meta[k]!=v for k,v in expected.items()):raise ValueError('Publisher CSV identities disagree')
    return dict(status='REGISTERED_BASIC',sequence_id=name,video_url=url,music_id=music,dancer_id=dancer,
        recording_uid='aist_capture:'+name,environment=auth['mapping'][name],publisher_metadata=meta,
        choreography_uid='aist_basic:'+genre+':'+ch,choreography_title=auth['table'][(genre,ch)],
        identity_scope='publisher genre-namespaced basic choreography; not unique generic movement title',
        group=row['group_uid'],role=row['purpose_role'])

def ncc_valid(reference,query):
    x=np.asarray(reference,float);y=np.asarray(query,float)
    if len(x)<len(y) or len(y)<2:raise ValueError('Insufficient real overlap')
    y=y-y.mean();energy=float(y@y)
    if energy<1e-10:raise ValueError('Silent alignment query')
    n=len(y);s=np.r_[0,np.cumsum(x)];ss=np.r_[0,np.cumsum(x*x)]
    variance=np.maximum(ss[n:]-ss[:-n]-(s[n:]-s[:-n])**2/n,1e-20)
    return correlate(x,y,mode='valid',method='fft')/np.sqrt(variance*energy)

def align_samples(reference,video,rate=8000,policy=POLICY):
    x=np.asarray(reference,float);y=np.asarray(video,float);whole=ncc_valid(x,y)
    best=int(whole.argmax());zero=float(whole[0])
    # Official refined video starts at canonical music origin. Prefix equality
    # can resolve a repeated-loop tie, but cannot rescue a poor prefix score.
    origin=0 if zero>=policy['minimum_ncc'] and float(whole[best])-zero<.005 else best
    scores=[];lags=[];anchors=[]
    length=min(len(y),round(policy['anchor_seconds']*rate))
    if length<2*rate:raise ValueError('Too little waveform for calibrated clock')
    for start in sorted(set([0,(len(y)-length)//2,len(y)-length])):
        center=origin+start;radius=round(policy['local_search_seconds']*rate)
        low=max(0,center-radius);high=min(len(x),center+length+radius)
        c=ncc_valid(x[low:high],y[start:start+length]);lag=low+int(c.argmax())-start
        lags.append(lag);scores.append(float(c.max()))
        anchors.append(dict(video_sample=start,source_lag_samples=lag,ncc=float(c.max())))
    lag=int(round(float(np.median(lags))));residual=max(abs(v-lag) for v in lags)/rate
    alternatives=np.flatnonzero(whole>=float(whole[best])-.005)
    distant=bool(np.any(abs(alternatives-origin)>rate*.25))
    valid=(min(scores)>=policy['minimum_ncc'] and residual<=policy['maximum_residual_seconds']
           and (max(lags)-min(lags))/rate<=policy['maximum_lag_spread_seconds']
           and (origin==0 or not distant))
    return dict(verified=bool(valid),audio_offset_seconds=lag/rate,anchors=anchors,
        minimum_ncc=min(scores),global_ncc=float(whole[best]),zero_ncc=zero,
        maximum_residual_seconds=residual,periodic_alternatives=distant,
        origin_anchor_used=origin==0,rate=rate,method='official_refined_origin_plus_three_anchor_waveform_ncc')

def calibrate(video,audio,native_frames):
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)]))
    streams={s['codec_type']:s for s in probe['streams'] if s['codec_type'] in ('video','audio')}
    if set(streams)!=set(('video','audio')):raise ValueError('Refined movie missing stream')
    pcm=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(video),'-map','0:a:0',
        '-ac','1','-ar','8000','-f','f32le','pipe:1'],timeout=60)
    y=np.frombuffer(pcm,dtype='<f4').astype(float)
    x,sr=sf.read(audio,always_2d=True);x=x.mean(1)
    from math import gcd
    divisor=gcd(sr,8000);x=resample_poly(x,8000//divisor,sr//divisor)
    motion_seconds=native_frames/60
    duration_error=abs(len(y)/8000-motion_seconds)
    # Validate exact annotation clock; no fitted time stretch is introduced.
    result=align_samples(x,y[:min(len(y),round(motion_seconds*8000))])
    result.update(motion_seconds=motion_seconds,decoded_audio_seconds=len(y)/8000,
        duration_error_seconds=duration_error,stream_metadata=streams)
    result['verified']=result['verified'] and duration_error<=POLICY['maximum_clock_duration_error_seconds']
    return result
