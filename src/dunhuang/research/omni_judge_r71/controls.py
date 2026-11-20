"""Frozen audio interventions; never optimize or change the actual motion."""
from pathlib import Path
import json,subprocess,hashlib
ROOT=Path('outputs/omni_judge_r71_01')

def wrong_music(current_sha,audio_files):
 candidates=sorted((p for p in audio_files.values() if p['raw_audio_sha256']!=current_sha),key=lambda p:p['path'])
 if not candidates:raise ValueError('No genuinely different registered music available')
 return candidates[0]
def run():
 jobs=json.loads((ROOT/'jobs.json').read_text());out=ROOT/'media_controls_v2';out.mkdir(exist_ok=False);result=[]
 source=Path('outputs/retiming_baselines_r69_01/main');cases=json.loads((source/'population.json').read_text())['cases'];by={r['pair_id']:r for r in json.loads((ROOT/'registered_pairs.json').read_text())};audio_files=json.loads((source/'viewer/protocol.json').read_text())['audio_files']
 for i,job in enumerate(jobs):
  for kind in ('muted','shift500ms','wrong_music'):
   row=dict(job,condition=kind)
   if kind!='muted':
    target=out/(job['pair_id']+'_'+job['order']+'_'+kind+'.mp4');command=['ffmpeg','-v','error','-nostdin','-i',job['video']]
    if kind=='shift500ms':command+=['-filter_complex','[0:a]adelay=500:all=1,atrim=duration=17.6[a]','-map','0:v:0','-map','[a]']
    else:
     current=cases[by[job['pair_id']]['case_index']]['audio_sha256'];donor=wrong_music(current,audio_files);audio=source/'viewer'/donor['path'];assert hashlib.sha256(audio.read_bytes()).hexdigest()==donor['sha256'];command+=['-i',str(audio),'-map','0:v:0','-map','1:a:0'];row['audio_donor']=donor;row['original_raw_audio_sha256']=current
    command+=['-t','17.6','-c:v','copy','-c:a','aac','-map_metadata','-1',str(target)];subprocess.run(command,check=True)
    row.update(video=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
   result.append(row)
 (ROOT/'jobs_controls.json').write_text(json.dumps(result,indent=2));print('CONTROLS_REGISTERED',len(result))
if __name__=='__main__':run()
