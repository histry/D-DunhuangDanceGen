"""Reuse the tested clock/camera; make new conditions and unknown world reference explicit."""
import argparse
from pathlib import Path
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.reperformance_r44.view import build as parent_view

def build(main,out):
    parent_view(main,out);out=Path(out);html=(out/'index.html').read_text(encoding='utf8').replace('R44','R45').replace('来源姿态不变，随音乐改变演法时间','同一来源动作，观察音乐怎样改变阶段安排').replace('Geometry / E1-W / C²平滑重定时 / 冻结R43失败样本 · 同一音乐时钟','Geometry / 冻结R44 C² / R45 Phase / 冻结R43 Core · 同一音乐时钟').replace('WrongMapB 使用B音乐求出的时间图；FixedMapMusicB与C2动作逐元素一致。','WrongMusic使用B音乐评分；NoMusic忽略声学输入；FixedMapMusicB与Phase动作逐元素一致。').replace('两侧Core调整时钟','两侧Core在固定来源可行图集合中选择时钟')
    html=html.replace('种子一／二中的几何输出相同，不是独立重复实验。','种子一／二的身体轨迹相同；压力案例的模型Root与接触通道可不同，不能视为独立来源重复。');html=html.replace('</main>','<p>Phase的动态约束直接用于候选筛查，不是独立物理通过证据。MatchedLinear/MatchedC2使用完全相同锚点，只改变时间插值。RMS与腕部加速度对齐均为代理，非文化或语义真值。</p></main>');(out/'index.html').write_text(html,encoding='utf8');js=(out/'viewer.js').read_text(encoding='utf8').replace('待生成区 [104,152)','固定Geometry桥 [104,152)')
    js=js.replace("const local=$('local').checked,origin=","if(p.reference&&!$('local').checked){x.fillStyle='#385a4b';x.font='14px system-ui';x.fillText('来源世界Root未知，请切换局部姿态',20,170);continue;}const local=$('local').checked,origin=")
    (out/'viewer.js').write_text(js,encoding='utf8');p=read(out/'protocol.json');p.update(release='R45',files=[dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.iterdir()) if f.name!='protocol.json']);write(out/'protocol.json',p)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--out',required=True);a=p.parse_args();build(a.main,a.out)
