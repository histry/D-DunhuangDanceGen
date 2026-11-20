"""Text-only official AWQ pilot. Machine opinions never become human records."""
import json,os,sys,time,subprocess,hashlib
from pathlib import Path
ROOT=Path('outputs/omni_judge_r71_01').resolve()
PROMPT='''你将看到同一音乐下并排的两段骨架动作，左为A，右为B。忽略外观。分别比较音乐协调、自然性、转场时机、连续性、幅度与动态丰富度。不得只因速度快或拍点密就判好；看不出差异允许TIE，缺少必要音频或视觉证据必须UNSURE。不要判断敦煌文化真实性。只输出JSON，字段 music_coordination,naturalness,transition_timing,continuity,richness，各字段为{preference:A或B或TIE或UNSURE,timestamps_seconds:[],reason:简短实际证据}。不要虚构时间点。'''
DIMS=['music_coordination','naturalness','transition_timing','continuity','richness']
def parse(raw):
 try:
  obj=json.loads(raw[raw.index('{'):raw.rindex('}')+1]);assert set(obj)==set(DIMS)
  for item in obj.values():
   assert set(item)=={'preference','timestamps_seconds','reason'} and isinstance(item['timestamps_seconds'],list)
   assert item['preference'] in ('A','B','TIE','UNSURE') and isinstance(item['reason'],str)
   assert all(isinstance(t,(int,float)) and not isinstance(t,bool) and 0<=t<17.6 for t in item['timestamps_seconds'])
  return obj
 except (ValueError,KeyError,AssertionError,TypeError):return None
def run(replay=False,full=False,controls=False):
 import torch
 from transformers import Qwen2_5OmniProcessor
 from qwen_omni_utils import process_mm_info
 repo=ROOT/'assets/Qwen2.5-Omni';commit=(ROOT/'receipts/official_commit.txt').read_text().strip()
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==commit
 modelpath=ROOT/'assets/model';jobs=json.loads((ROOT/('jobs_full.json' if full else 'jobs_controls.json' if controls else 'jobs.json')).read_text())
 output=ROOT/('pilot_replay' if replay else 'full' if full else 'controls' if controls else 'pilot');output.mkdir(exist_ok=False)
 os.chdir(repo/'low-VRAM-mode');sys.path.insert(0,str(Path.cwd()));namespace={'__name__':'official_awq_adapter'}
 prefix=Path('low_VRAM_demo_awq.py').read_text().split('device_map =')[0]
 exec(compile(prefix,'official_low_VRAM_demo_awq_prefix','exec'),namespace)
 cls=namespace['Qwen2_5_OmniAWQForConditionalGeneration'];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 model=cls.from_quantized(str(modelpath),model_type='qwen2_5_omni',torch_dtype=torch.float16,attn_implementation='sdpa')
 model.model.disable_talker()
 load_peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
 processor=Qwen2_5OmniProcessor.from_pretrained(modelpath)
 rows=[]
 for index,job in enumerate(jobs[:1] if replay else jobs):
  video=Path(job['video'])
  if not video.is_absolute():video=ROOT.parent.parent/video
  assert hashlib.sha256(video.read_bytes()).hexdigest()==job['sha256']
  message=[dict(role='system',content=[dict(type='text',text='You are an audio-video comparison assistant. Report uncertainty; automated opinions are not ground truth.')]),dict(role='user',content=[dict(type='video',video=str(video),fps=1.0,max_pixels=160*28*28),dict(type='text',text=PROMPT)])]
  text=processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
  use_audio=job['condition']!='muted'
  audios,images,videos=process_mm_info(message,use_audio_in_video=use_audio)
  inputs=processor(text=text,audio=audios,images=images,videos=videos,return_tensors='pt',padding=True,use_audio_in_video=use_audio,fps=1.0)
  shapes={k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')}
  input_hashes={k:hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in inputs.items() if torch.is_tensor(v)}
  grid=inputs.get('video_second_per_grid',inputs.get('second_per_grid_ts'))
  if hasattr(grid,'tolist'):grid=grid.tolist()
  if not videos or use_audio and not audios:raise ValueError('Required modalities did not enter the processor')
  torch.manual_seed(71071);torch.cuda.manual_seed_all(71071);torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
  inputs=inputs.to('cuda')
  with torch.inference_mode():ids=model.generate(**inputs,use_audio_in_video=use_audio,return_audio=False,do_sample=False,max_new_tokens=512)
  if isinstance(ids,tuple):raise ValueError('Text-only inference unexpectedly returned audio tuple')
  raw=processor.batch_decode(ids[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
  row=dict(job=job,model_revision=json.loads((ROOT/'receipts/model_registration.json').read_text())['revision'],official_code_commit=commit,prompt=PROMPT,
   sampled_video_shapes=[list(v.shape) for v in videos],audio_sample_shapes=[list(v.shape) for v in (audios or [])],processor_shapes=shapes,processor_tensor_sha256=input_hashes,video_seconds_per_grid=grid,use_audio_in_video=use_audio,
   requested_fps=1,seed=71071,attention_implementation='sdpa',gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,load_peak_bytes=load_peak,raw_response=raw,parsed=parse(raw),seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
   automatic_judgment=True,human_reviewed=False,human_agreement='NOT_EVALUABLE',Native24='NOT_EVALUABLE')
  (output/(str(index)+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2));rows.append(row);print('R71_INFERENCE',index+1,len(jobs),row['peak_allocated_bytes'],row['parsed'] is not None,flush=True)
 (output/'summary.json').write_text(json.dumps(dict(inferences=len(rows),structured=sum(r['parsed'] is not None for r in rows),human_reviewed=False,quality_winner=None,seconds_including_load=time.perf_counter()-start),indent=2))
 if replay:
  previous=json.loads((ROOT/'pilot/0.json').read_text());current=rows[0]
  fields=['job','model_revision','official_code_commit','prompt','sampled_video_shapes','audio_sample_shapes','processor_shapes','processor_tensor_sha256','video_seconds_per_grid','use_audio_in_video','raw_response','parsed']
  differences=[f for f in fields if previous[f]!=current[f]]
  (output/'independent_replay.json').write_text(json.dumps(dict(fresh_process=True,compared_fields=fields,differences=differences,exact=not differences,optimizer_calls=0,human_reviewed=False),indent=2))
if __name__=='__main__':run('--replay' in sys.argv,'--full' in sys.argv,'--controls' in sys.argv)
