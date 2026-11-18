"""Freeze measured R50 outcomes and verify them in a fresh process."""
from pathlib import Path
import subprocess
import numpy as np
from dunhuang.research.complete_r36.core import read,write
from dunhuang.research.reperformance_r44.replay import digest
from .common import ROOT

PUB = Path("outputs/r50_publication")
ROOTS = [ROOT,Path("outputs/transition_planning_r50_replay"),Path("outputs/attribution_r50_01"),
         Path("outputs/attribution_r50_replay")]


def run():
    from .conditions import diagnose
    # Re-execute current condition contract, including exact protected-frame permissions.
    diagnose()
    p = read(ROOT/"result.json")
    a = read("outputs/attribution_r50_01/result.json")
    ind = read(PUB/"independent.json")
    assert ind["exact_arrays_and_scores"]
    acceptance = read(ROOT/"browser_receipt.json")
    assert acceptance["protocol_sha256"]==digest(ROOT/"viewer/protocol.json")
    assert acceptance["verified_unix"]<Path("outputs/attribution_r50_01/result.json").stat().st_mtime
    summary = dict(plan_cases=[{k:v for k,v in r.items() if k!="connections"} for r in p["cases"]],
                   attribution=a["summary"],condition_probe=a["condition_probe"],independent=ind,
                   pytest=(PUB/"pytest.log").read_text().strip().splitlines()[-1],
                   model=a["model"],can_route=False,neural_optimizer_steps=0,
                   code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    write(PUB/"summary.json",summary)
    good = [r for r in p["cases"] if r["status"]=="NUMERICAL_SHORT_PLAN"]
    s = a["summary"]
    lines = ["# R50 显式过渡预算与完整音乐归因","",
             f"固定人口18组短编舞中，{len(good)}组通过既定身体相对坐标数值筛查。"
             "全曲中的目标区间显式包含Core与Transition，全部来源Core保留原区间。"
             "主效果实际验收之后运行音乐归因和独立重算。","",
             f"完整主评分换音乐改变 {s['full_music_top_changes']}/{s['slots']} 个首选，"
             f"Local换音乐改变 {s['local_music_top_changes']}/{s['slots']}。"
             f"正确／错音乐的完整分数排名Spearman中位数为 {s['median_spearman']:.6f}。"
             "这些变化是归因诊断，真实匹配收益继续参考历史Val的配对区间。","",
             "## 观看入口","",
             "本机 outputs/transition_planning_r50_01/viewer/index.html 提供三首音乐各两事件、三事件共六段。"
             "48帧过渡在时间轴中真实占位，两视角共享同一保存轨迹和音乐。"
             "Core为真实ChangE重定时，连接为固定G0/G1几何提案。","",
             "## 时间与接法","",
             "|案例|Core帧|Transition帧|总帧|资格|全片D3 Q95|",
             "|---|---:|---:|---:|---|---:|"]
    for r in p["cases"]:
        lines.append(f"|{r['key']}|{r['core_frames']}|{r['bridge_frames']}|{r['total_frames']}|{r['status']}|{r.get('whole_clip_D3')}|")
    lines += ["","每首音乐按时长中心放置实际区间，事件时长240帧，连接24/48/72帧。"
              "每来源取元数据三分位两条完整候选，每Core保留四个不同来源候选，在连接资格下联合搜索。"
              "借用R48 Train来源D2/D3/肩腕参考筛查桥接，保留几何图域和实际差分指标。"
              "全片动态另存，不以桥接通过代替内部质量。世界Root、接触和完整Native24资格仍缺观测。",
              "","## 评分项与来源偏置","","|来源|候选数|完整正确减错音乐均值|标准差|","|---|---:|---:|---:|"]
    for uid,v in s["source_delta"].items():
        lines.append(f"|{uid}|{v['candidates']}|{v['mean']}|{v['std']}|")
    lines += ["","每个候选保留原R49 A0、共同支持上的Local、Multi、Warp，完整正确/错音乐及同歌近能量错时分数。"
              "音频置换保持候选、来源时间图与Token掩码一致。"
              "另保存A1/A2/A3、等来源元数据子池、Top-K、Top1/2间隔和评分项方差。"
              "真实未知敦煌音乐的艺术适配标签仍待建立。历史Val沿用R49固定结果，五偏移Content25.46%、"
              "换音乐21.20%，配对95%区间跨零。新盲测尚未运行。","",
              "## 条件传播","","独立PhaseReference子类修复可变长来源路径的阶段置零，保留旧模块供冻结调用。"
              "MusicReperformanceCondition显式校验音乐、来源特征、源时钟、阶段、双侧上下文、掩码和编辑许可。"
              "固定随机权重与非零读出用于检查结构传播，尚未加载真实生成权重验证训练收益。"
              f"参考/查询阶段梯度范数 {a['condition_probe']['reference_and_query_phase_gradient_norm']}，"
              f"阶段置零输出RMS变化 {a['condition_probe']['phase_zero_output_rms']}。"
              "无效来源改成NaN后输出逐位一致，禁编辑首尾区域精确保留输入隐藏状态。"
              "主归因复验完成后补充的编辑许可保护在发布进程重新执行断言。音乐和上下文继续由现有主干与Controller路由，"
              "本轮结构探针不作其质量改进结论。","",
              "## 验证与发布","",f"完整回归 {summary['pytest']}。"
              "主轨迹与完整归因均由新进程重算，数组及科学JSON精确一致，R49的17个封存文件逐哈希保持。"
              "代码先GitHub，服务器tmux运行，观看资源按协议逐SHA回传。"
              "原始轨迹与详尽评分保留服务器，公开汇总和观看页回传本机。"
              "本轮神经训练0，完整Native通过未建立，研究路由继续Shadow。"]
    (PUB/"report.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    files = []
    for root in ROOTS:
        for f in sorted(root.rglob("*")):
            if f.is_file():
                files.append(dict(path=str(f),bytes=f.stat().st_size,sha256=digest(f)))
    for name in ("summary.json","report.md","independent.json"):
        f=PUB/name
        files.append(dict(path=str(f),bytes=f.stat().st_size,sha256=digest(f)))
    write(PUB/"seal.json",dict(files=files))


def recheck():
    files = read(PUB/"seal.json")["files"]
    for f in files:
        p = Path(f["path"])
        assert p.stat().st_size==f["bytes"] and digest(p)==f["sha256"],str(p)
    write(PUB/"seal_recheck.json",dict(files=len(files),all_hashes_verified=True,
          commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()))
