"""One frozen short-task repair; global attempt ledger caps all GPU calls at48."""
import os,json,sys,time,hashlib,subprocess
from pathlib import Path
import numpy as np
from dunhuang.research.local_timing_r68.common import read,write,sha
ROOT=Path('outputs/judge_repair_r72_01').resolve();ASSETS=Path('outputs/omni_judge_r71_01/assets').resolve()
MUSIC='两侧同步播放同一段8秒视频与共同音轨，左A右B。只比较0.5至8秒内运动速度峰与可听音乐重音的同步程度。相当TIE表示有足够声音和运动证据且两侧相当；如果没有实际听到音轨或证据不足必须UNSURE，不得只根据画面猜音乐。仅输出一个短JSON：{"preference":"A或B或TIE或UNSURE"}。不要解释，不要时间戳。'
NATURAL='两侧同镜头同步播放8秒骨架。比较动作自然性，允许TIE或UNSURE，不假定运动越快越好或静止一定不好。同时判断两边实际运动是否明显不同。只输出短JSON：{"preference":"A或B或TIE或UNSURE","different":true或false或null}。不要解释，不要时间戳。'
LOCALIZE='8秒片段中仅有一次明显的手臂构形展开峰。请指出它最明显的时刻，秒数必须在0至8以内；无法确定填null。只输出JSON：{"seconds":数字或null}，不要解释、区间或其他时间点。'

def parse(raw,task):
    try:
        x=json.loads(raw[raw.index('{'):raw.rindex('}')+1])
        if task=='time':
            assert set(x)=={'seconds'};v=x['seconds'];assert v is None or isinstance(v,(float,int)) and not isinstance(v,bool) and 0<=v<8
        else:
            assert set(x)==({'preference','different'} if task=='naturalness' else {'preference'})
            assert x['preference'] in ['A','B','TIE','UNSURE']
            if task=='naturalness':assert x['different'] is None or type(x['different']) is bool
        return x
    except (ValueError,AssertionError,TypeError,KeyError):return None

def inputs_for(processor,job):
    import torch,soundfile as sf
    from torchvision.io import read_video
    from torchvision.transforms.functional import resize
    path=Path(job['video']);path=path if path.is_absolute() else ROOT.parent.parent/path
    assert sha(path)==job['video_sha256']
    V,_,info=read_video(str(path),pts_unit='sec',output_format='TCHW');assert len(V)==240 and info['video_fps']==30
    indices=torch.arange(0,240,30//job['fps']);V=resize(V[indices].float(),[252,448],antialias=True)
    audio=None
    if job['use_audio']:
        p=Path(job['audio']);p=p if p.is_absolute() else ROOT.parent.parent/p;assert sha(p)==job['audio_sha256'];y,sr=sf.read(p,dtype='float32');assert sr==16000 and y.shape==(128000,);audio=[y]
    prompt=LOCALIZE if job['task']=='time' else NATURAL if job['task']=='naturalness' else MUSIC
    messages=[dict(role='system',content=[dict(type='text',text='You compare only evidence actually provided. Automated diagnostic answers are not human labels.')]),dict(role='user',content=[dict(type='video',video=str(path)),dict(type='text',text=prompt)])]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=text,audio=audio,images=None,videos=[V],return_tensors='pt',padding=True,use_audio_in_video=job['use_audio'],fps=float(job['fps']))
    assert ('input_features' in inputs)==job['use_audio']
    grid=inputs['video_second_per_grid'].tolist();assert abs(grid[0]*int(inputs['video_grid_thw'][0,0])-8)<1e-6
    meta=dict(prompt=prompt,sampled_indices=indices.tolist(),sampled_times_seconds=(indices/30).tolist(),requested_and_actual_fps=job['fps'],
        time_grid=grid,input_shapes={k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')},
        tensor_sha256={k:hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in inputs.items() if torch.is_tensor(v)},
        audio_present='input_features' in inputs,video_reader='TORCHVISION_EXACT_REGULAR_FRAME_SAMPLING_NO_LINSPACE_DRIFT')
    return inputs,meta

def run(phase='dev'):
    import torch
    from transformers import Qwen2_5OmniProcessor
    jobs=read(ROOT/'jobs.json');jobs=[j for j in jobs if j['role']==('dev' if phase in ['dev','replay'] else 'validation')]
    if phase=='validation':assert read(ROOT/'dev/summary.json')['development_2fps_gate_passed']
    if phase=='replay':jobs=jobs[:1]
    out=ROOT/phase;out.mkdir(exist_ok=False)
    processor=Qwen2_5OmniProcessor.from_pretrained(ASSETS/'model',local_files_only=True)
    # Real integration preflight before loading quantized model weights.
    for j in jobs:inputs_for(processor,j)
    repo=ASSETS/'Qwen2.5-Omni';assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()=='d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42'
    os.chdir(repo/'low-VRAM-mode');sys.path.insert(0,str(Path.cwd()));ns={'__name__':'frozen_official_awq'}
    exec(compile(Path('low_VRAM_demo_awq.py').read_text(encoding='utf-8').split('device_map =')[0],'official_awq','exec'),ns)
    model=ns['Qwen2_5_OmniAWQForConditionalGeneration'].from_quantized(str(ASSETS/'model'),model_type='qwen2_5_omni',torch_dtype=torch.float16,attn_implementation='sdpa')
    model.model.disable_talker();model.model.thinker.to('cuda');model.model.eval();rows=[]
    ledger=ROOT/'generation_attempts.jsonl'
    for job in jobs:
        inputs,meta=inputs_for(processor,job);attempts=ledger.read_text().splitlines() if ledger.exists() else []
        if len(attempts)>=48:raise RuntimeError('Hard generation budget exhausted')
        with ledger.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(attempt=len(attempts)+1,phase=phase,job_id=job['job_id']))+'\n');f.flush();os.fsync(f.fileno())
        torch.manual_seed(72073);torch.cuda.manual_seed_all(72073);torch.cuda.reset_peak_memory_stats();start=time.perf_counter();limit=64 if job['task']=='time' else 128
        row=dict(job=job,input=meta,human_reviewed=False,automatic=True,quality_truth='NOT_EVALUABLE')
        try:
            inputs=inputs.to('cuda')
            with torch.inference_mode():result=model.model.thinker.generate(**inputs,use_audio_in_video=job['use_audio'],do_sample=False,max_new_tokens=limit,return_dict_in_generate=True)
            ids=result.sequences;new=ids[0,inputs['input_ids'].shape[1]:];raw=processor.decode(new,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            eos=model.model.thinker.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
            stop='EOS' if len(new) and int(new[-1]) in eos else 'LENGTH' if len(new)>=limit else 'OTHER'
            row.update(raw_response=raw,parsed=parse(raw,job['task']),generated_tokens=len(new),token_limit=limit,last_token=int(new[-1]) if len(new) else None,stop_reason=stop)
        except Exception as e:row.update(error=repr(e),parsed=None);torch.cuda.empty_cache()
        row.update(seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated());write(out/(job['job_id']+'.json'),row);rows.append(row);print('R72_CALL',phase,len(rows),len(jobs),row.get('parsed'),flush=True)
    selected=[r for r in rows if r['job']['fps']==2]
    def correct(r):
        p=r.get('parsed');t=r['job']['test']
        if p is None:return False
        if t=='Different':return p['different'] is True
        if t=='Localization':return p['seconds'] is not None and abs(p['seconds']-4)<=.5
        return p['preference']==r['job']['expected'] and (t!='Identical' or p['different'] is False)
    summary=dict(calls=len(rows),structured=sum(r.get('parsed') is not None for r in rows),primary_2fps_correct=sum(correct(r) for r in selected),primary_2fps_total=len(selected),
        development_2fps_gate_passed=phase=='dev' and all(correct(r) for r in selected),human_agreement='NOT_EVALUABLE',R69_bulk_ranking='NOT_AUTHORIZED_BY_DIAGNOSTIC_PASS',human_reviewed=False)
    write(out/'summary.json',summary)
    if phase=='replay':
        old=read(ROOT/'dev'/(jobs[0]['job_id']+'.json'));fields=['input','raw_response','parsed','generated_tokens','stop_reason'];diff=[k for k in fields if old.get(k)!=rows[0].get(k)]
        write(out/'independent.json',dict(fresh_process=True,exact=not diff,differences=diff,generation_calls=1,budget_includes_replay=True))

if __name__=='__main__':run('validation' if '--validation' in sys.argv else 'replay' if '--replay' in sys.argv else 'dev')
