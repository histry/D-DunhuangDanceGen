"""Common camera/audio-clock delivery of actual G0/G1/G2 arrays."""
import argparse
from pathlib import Path
from dunhuang.research.bridge_audit_r46.view import run as parent_view
from dunhuang.research.complete_r36.core import read,write,sha

def run(main,out):
    parent_view(main,out);out=Path(out);s=(out/'index.html').read_text(encoding='utf8').replace('R46','R47').replace('来源为什么静止？几何桥在第117帧做了什么？','相同观察端点：冻结五次桥与分段 SO(3) 桥').replace('冻结Geometry／仅观察端点重建／冻结Phase · 共同真实音乐时钟','G0历史五次桥／G1重新锚定桥／G2流形样条对照').replace('ObservedRebuild仅重建身体旋转，Root保持原模型先验；原始缺口只在连续录制中有真值。Oracle页面仅含Train诊断上限，不是可部署生成器。','G1直接使用两端观测的身体角速度/角加速度，逐段检查局部图域。保护区与Root/contact逐位保留；几何合法不等于自然或物理合格。').replace('来源容量是实测阶段与合法时钟范围，不是艺术表现能力。Oracle使用Train评价代理事后选图，不可进入正式推理。','端点相同不代表隐藏动作相同。仅同录制缺口有重建真值；异源连接自然性没有人工评价，不把几何合法当作Native通过。');(out/'index.html').write_text(s,encoding='utf8');p=read(out/'protocol.json');p.update(release='R47',files=[dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.iterdir()) if f.name!='protocol.json']);write(out/'protocol.json',p)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.main,a.out)
