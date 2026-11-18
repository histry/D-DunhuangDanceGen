"""Shared actual-motion viewer; explicit source/bridge responsibility and oracle labels."""
import argparse
from pathlib import Path
from dunhuang.research.phase_r45.view import build as parent_view
from dunhuang.research.complete_r36.core import read,write,sha

def run(main,out):
    parent_view(main,out);out=Path(out);html=(out/'index.html').read_text(encoding='utf8');html=html.replace('R45','R46').replace('同一来源动作，观察音乐怎样改变阶段安排','来源为什么静止？几何桥在第117帧做了什么？').replace('Geometry / 冻结R44 C² / R46 Phase / 冻结R43 Core · 同一音乐时钟','冻结Geometry／仅观察端点重建／冻结Phase · 共同真实音乐时钟')
    html=html.replace('</header>','<button id="frame117">跳到第117帧</button><p>本轮为责任审计：没有新增神经动作。ObservedRebuild仅重建身体旋转，Root保持原模型先验；原始缺口只在连续录制中有真值。Oracle页面仅含Train诊断上限，不是可部署生成器。</p></header>')
    # Avoid carrying inherited claims about methods that this page does not contain.
    html=html.replace('WrongMusic使用B音乐评分；NoMusic忽略声学输入；FixedMapMusicB与Phase动作逐元素一致。试听只切换音乐，不重新生成动作。','点击同一帧比较来源、冻结几何桥与独立重建；误差与候选容量在来源和指标详情中。')
    html=html.replace('Phase的动态约束直接用于候选筛查，不是独立物理通过证据。MatchedLinear/MatchedC2使用完全相同锚点，只改变时间插值。RMS与腕部加速度对齐均为代理，非文化或语义真值。','来源容量是实测阶段与合法时钟范围，不是艺术表现能力。Oracle使用Train评价代理事后选图，不可进入正式推理。')
    js=(out/'viewer.js').read_text(encoding='utf8');js=js.replace("$('play').onclick=toggle;","$('frame117').onclick=()=>{audio.pause();playing=false;seek(117);};$('play').onclick=toggle;")
    (out/'index.html').write_text(html,encoding='utf8');(out/'viewer.js').write_text(js,encoding='utf8');p=read(out/'protocol.json');p.update(release='R46',files=[dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.iterdir()) if f.name!='protocol.json']);write(out/'protocol.json',p)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.main,a.out)
