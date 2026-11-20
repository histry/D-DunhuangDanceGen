"""Publish scalar/automatic R71 evidence. Never publish model weights or AV data."""
import hashlib,json,re,shutil
from collections import Counter
from pathlib import Path

ROOT=Path('outputs/omni_judge_r71_01')
OUT=Path('reports/r71_release')
SCIENCE='d5d92610d73a443c0c7677a841cada6c36a26fb9'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_bytes((json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf-8'))

def main():
    pilot=[read(ROOT/'pilot'/f'{i}.json') for i in range(6)]
    controls=[read(ROOT/'controls'/f'{i}.json') for i in range(18)]
    for stage in ['pilot','replay','controls']:
        assert read(ROOT/'receipts'/(stage+'_exit.json'))['exit_status']==0
    assert all(read(ROOT/'receipts'/('input_'+s+'.json'))['passed'] for s in ['pilot','replay','controls'])
    replay=read(ROOT/'pilot_replay/independent_replay.json')
    motion=read(ROOT/'receipts/independent_motion_media_replay.json')
    assert motion['trajectories']==69 and motion['motion_differences']==[]
    muted=[r for r in controls if r['job']['condition']=='muted']
    assert len(muted)==6 and all(not r['use_audio_in_video'] and not r['audio_sample_shapes'] and 'input_features' not in r['processor_shapes'] for r in muted)
    noaudio=Counter(r['parsed']['music_coordination']['preference'] if r['parsed'] else 'INVALID_SCHEMA' for r in muted)
    badtime=[]
    for i,r in enumerate(controls):
        intervals=re.findall(r'"(\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?)"',r['raw_response'])
        bad=[s for s in intervals if not 0<=float(s.split('-')[0])<float(s.split('-')[1])<=17.6]
        if bad:badtime.append(dict(control_index=i,condition=r['job']['condition'],raw_interval_strings=bad))
    reasons=[]
    if noaudio.get('UNSURE',0)!=6:reasons.append('MISSING_AUDIO_ABSTENTION_CONTRACT_NOT_MET')
    if any(r['parsed'] is None for r in controls):reasons.append('CONTROL_OUTPUT_SCHEMA_FAILURES')
    if badtime:reasons.append('RAW_TIME_EVIDENCE_OUTSIDE_VIDEO_SUPPORT')
    if not replay['exact']:reasons.append('FRESH_PROCESS_REPLY_DIFFERENCES')
    assert reasons,'Review a passed reliability gate before publishing an unqualified result'
    qualified=dict(scientific_code_commit=SCIENCE,registered_pairs=60,eligible_pairs=56,excluded_pairs=4,
        prepared_full_ordered_comparisons=112,executed_full_ordered_comparisons=0,
        pilot_inferences=6,pilot_structured=sum(r['parsed'] is not None for r in pilot),
        control_inferences=18,control_structured=sum(r['parsed'] is not None for r in controls),
        fresh_process_replay_inferences=1,fresh_process_replay_exact=replay['exact'],
        independent_motion_rows=69,independent_motion_differences=0,
        muted_music_response_counts=dict(noaudio),controls_with_out_of_support_raw_intervals=len(badtime),
        reliability_reasons=reasons,automatic_judge_can_rank_quality=False,can_route=False,
        full112_status='PREPARED_NOT_EXECUTED_RELIABILITY_GATE_NOT_MET',
        human_reviewed=False,valid_human_records=0,human_agreement='NOT_EVALUABLE',artistic_quality='NOT_EVALUABLE',
        Native24='NOT_EVALUABLE',MotionCritic='INPUT_CONTRACT_NOT_ADMITTED',new_training_steps=0,quality_winner=None,
        scope='FIXED_QUANTIZED_MODEL_PROMPT_SPARSE_BODY22_SKELETON_PRESET; NOT_A_GLOBAL_MODEL_VERDICT')
    OUT.mkdir(exist_ok=False)
    write(OUT/'qualification.json',qualified);write(OUT/'out_of_support_raw_intervals.json',badtime)
    names=['automatic_summary.json','registered_pairs.json','media_manifest.json','media_manifest_full.json',
        'receipts/model_registration.json','receipts/model_files.json','receipts/pip_freeze.txt','receipts/official_commit.txt',
        'receipts/independent_motion_media_replay.json','receipts/independent_audio_controls.json',
        'receipts/input_pilot.json','receipts/input_replay.json','receipts/input_controls.json',
        'receipts/pilot_exit.json','receipts/replay_exit.json','receipts/controls_exit.json',
        'receipts/browser_acceptance.json','receipts/final_delivery.json','receipts/code_transport_final.json',
        'pilot/summary.json','pilot_replay/summary.json','pilot_replay/independent_replay.json','controls/summary.json',
        'receipts/response_diagnostics.json','receipts/pair_identity_audit.json','receipts/ssh_cleanup.json']
    names += [f'pilot/{i}.json' for i in range(6)]+['pilot_replay/0.json']+[f'controls/{i}.json' for i in range(18)]
    for name in names:
        source=ROOT/name
        assert source.is_file(),name
        target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    rows=pilot+controls
    peak=max(r['peak_allocated_bytes'] for r in rows)/2**30
    load=max(r['load_peak_bytes']['allocated'] for r in rows)/2**30
    document=f'''# R71 音视频自动评价先导与可靠性审查

结论：Qwen2.5-Omni-7B-AWQ 已在单张 RTX4090 上完成真实音视频推理与独立复验；当前固定评价配置尚不能作为可信的编舞质量排名器。没有产生人工评价真值或方法质量赢家。

## 冻结人口与真实执行

- 科学代码：`{SCIENCE}`。R69 登记60个E1-W对其他方法的配对，56个双方通过原G1门；4个排除位置保留。112个顺序互换输入已制作，本轮未扩大执行。
- 先导固定案例0、4、7，E1-W对Single64，AB/BA共6次；均为历史开发案例，三条来源的正确音乐均来自dunhuangwu3。完整R69人口涉及3首历史音乐、8个来源区间，不是独立新音乐测试。
- 先导6次、音频控制18次、新进程复验1次，共25次已保存推理。结构化有效数：先导{qualified['pilot_structured']}/6、控制{qualified['control_structured']}/18。原始回答及失败均保存，不重试挑选、不补写偏好。
- 新进程复验 `exact={replay['exact']}`；输入张量哈希、真实采样时钟、原始文本及解析结果逐项比较。运动学新进程复算69个已保存方法位置，差异0；这些不是69份独立录制。

## 输入与部署合同

官方Qwen低显存源码固定为 `d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42`。权重从作者发布的ModelScope镜像获取，20文件逐项固定版本和SHA256；依赖位于独立 `dancejudge_r71` 环境。没有训练或修改模型权重。

输出文本使用同一官方音视频Thinker，关闭语音生成；SDPA、torchvision视频读取、固定71071种子、贪心解码、512新增token预算。加载峰值分配显存约{load:.2f} GiB，推理峰值约{peak:.2f} GiB；同时保存原始字节数和reserved读数，不与官方理论最低值混作同条件比较。

每段528帧/30FPS/17.6秒，统一相机、骨架与比例。请求1FPS后，官方读取器实际保留16帧，采样率约0.909FPS；传入处理器标量真实采样率，双帧网格约2.2秒，并保存全部原始帧索引。该网格是稀疏近似，不能用于逐帧节拍或高频抖动真值。完整30FPS数组诊断独立报告。

不加载权重的真实处理器集成检查在GPU推理前执行。静音输入没有音频张量；异曲供体具有不同原WAV哈希；延迟500ms为补零、非循环移位。独立检查确认18个控制的视频码流不变及实际音频干预正确。

## 可靠性结果与边界

静音条件的音乐协调回答分布：`{json.dumps(dict(noaudio),ensure_ascii=False)}`。冻结提示要求缺少必要音频时选UNSURE，因此该合同尚未通过。不能仅凭相对偏好不变，就断言模型完全不使用音乐；特别是细微Warp差异可能本就缺乏可感知差别。

有{len(badtime)}条控制原文包含越过17.6秒支持范围的数字时间区间；详见逐条记录。部分输出也存在长度/结构问题，原文和再分词诊断单独保存。格式截断、时间证据越界和音乐识别能力不作同一原因解释。模型报告的区间保留为区间，不捏造中点标签。

本轮不继续扩大112次评价，不根据这些自动回答选出赢家或升级Planner。该结论只针对当前量化模型、提示词、稀疏骨架呈现和输出预算；不推广为所有多模态模型的结论。

人工有效记录为0，模型与人工一致性、独立音乐/艺术质量继续NOT_EVALUABLE。没有给缺失世界Root的身体22关节数据补零后套用MotionCritic；Native24仍NOT_EVALUABLE。相同源动作的几何/动态指标不等于敦煌文化真实性。

## 交付与复现

观看入口：`outputs/omni_judge_r71_01/index.html`，六段真实A/B视频已回传并验收；临时127.0.0.1服务已关闭。观看视频由R69冻结动作统一渲染。

`bash scripts/setup_omni_r71.sh` 安装隔离环境；在tmux中运行 `bash scripts/run_omni_r71.sh pilot`、`replay`、`controls`。本轮科学结果保持冻结；新运行应使用新的输出目录。`full`入口保留，当前未取得扩展评价资格。

服务器GitHub TLS中断后使用已发布提交的Git包校验部署，详情见传输回执；不关闭TLS校验。首版裁剪、语音包装器、ASCII落盘和FPS接口失败均留存，未并入本轮有效质量统计。临时SSH辅助包已按精确文件名和哈希清理；模型资产、科学输出和失败回执保留。

官方实现：[Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni)。R71采用其低显存实现与音视频处理接口，未声称复现TMD-Bench或取得MotionCritic原指标。
'''
    Path('reports/R71音视频自动评价先导与可靠性审查报告.md').write_bytes(document.encode('utf-8'))
    manifest={str(p.relative_to(OUT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file()}
    write(OUT/'manifest.json',dict(scientific_code_commit=SCIENCE,files=manifest))
    print(json.dumps(qualified,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
