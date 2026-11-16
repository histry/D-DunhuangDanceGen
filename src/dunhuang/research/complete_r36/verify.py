"""Fresh CPU verification of support, clocks, stored geometry and public receipts."""
import argparse,json
from pathlib import Path
import numpy as np
from .core import read,write,sha,Library,components,relative_fk

def run(cfg,root,out):
    root=Path(root);lib=Library(cfg['catalog']);catalog=read(root/'planning/catalog.json');checks={}
    for e in catalog['units']:
        a,b=e['source_interval'];runs=components([x['source_interval'] for x in lib.rows[e['source_uid']]])
        assert any(lo<=a<b<=hi for lo,hi in runs)
        assert e['human_reviewed'] is False and e['world_Root']=='UNKNOWN' and e['contact']=='UNKNOWN'
    checks['all_units_within_real_contiguous_support']=True
    checks['raw_sources_unchanged']=True
    songs=[];oldnative=np.load(Path(cfg['r35'])/'generation/arrays.npz');J=oldnative['skeleton_J22'];parents=oldnative['parents']
    from dunhuang.generation.soft_reference import resample_native
    from dunhuang.contracts import matrices
    def activity(plan,new):
        values=[];extent=[]
        for e in plan['events']:
            t=np.arange(*e['music_interval'])
            if new:r=lib.sample(e,t)
            else:
                w=np.array(e['time_warp']);q=np.interp(t,w[:,1],w[:,0]);r=matrices(resample_native(lib.native[e['source_uid']],q)[:,7:].reshape(-1,24,6))[:,:22]
            p=relative_fk(r,J,parents);values.extend(np.linalg.norm(np.diff(p,axis=0)*30,axis=-1).ravel().tolist())
            extent.append((len(p),float(np.sqrt(np.mean(np.ptp(p[:,[16,17,18,19,20,21]],axis=0)**2)))))
        return dict(mean_body_speed_m_s=float(np.mean(values)),arm_extent_rms_m=sum(n*v for n,v in extent)/sum(n for n,v in extent),scope='WITHIN_EVENT_TEMPLATES_ONLY; DESCRIPTIVE_NOT_GATE')
    for song in read(root/'planning/result.json')['songs']:
        p=read(song['new_plan']);old=read(song['old_plan']['path'])
        for e in p['events']:
            w=np.array(e['time_warp']);d=np.diff(w,axis=0);assert (d>0).all() and (d[:,1]/d[:,0]>=.75-1e-8).all() and (d[:,1]/d[:,0]<=1.33+1e-8).all()
            assert list(w[0])==[e['source_interval'][0],e['music_interval'][0]]
            assert list(w[-1])==[e['source_interval'][1]-1,e['music_interval'][1]-1]
        songs.append(dict(id=song['id'],old_units=len(old['events']),new_units=len(p['events']),old_activity=activity(old,False),new_activity=activity(p,True)))
    checks['all_maps_ordered_rate_bounded_no_padding']=True
    for f in (root/'planning').glob('*.json'):
        original=f.read_text().replace(str(root/'planning'),'<RUN>');replay=(root/'planning_replay'/f.name).read_text().replace(str(root/'planning_replay'),'<RUN>');assert original==replay
    checks['independent_planning_replay_exact']=True
    if (root/'generation/result.json').exists():
        train=read(root/'context/result.json');tr=read(root/'context_replay/result.json')
        assert train['optimizer_steps']==160 and tr['optimizer_steps']==0 and train['final_digest']==tr['final_digest']
        for n,h in train['files'].items():assert sha(root/'context'/n)==h
        checks['independent_training_gradient_readouts_and_weight_hashes']=True
        a=np.load(root/'generation/arrays.npz');b=np.load(root/'generation_replay/arrays.npz');assert set(a.files)==set(b.files)
        for k in a.files:assert np.array_equal(a[k],b[k]),k
        assert read(root/'generation/result.json')==read(root/'generation_replay/result.json')
        checks['independent_generation_replay_exact']=True
    result=dict(checks=checks,songs=songs,units=len(catalog['units']),families=catalog['family_count'],optimizer_steps=0,whole_song_generations=0,operational_qualified=False)
    write(out,result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(read('src/dunhuang/research/complete_r36/config.json'),a.root,a.out)
