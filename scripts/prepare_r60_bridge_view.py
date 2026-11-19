"""Replay actual saved geometric bridges on the preselected main sources."""
import json
import numpy as np
from dunhuang.research.phase_integrity_r60.common import ROOT,read,write,fresh,relative_fk,sha
from dunhuang.research.strong_baseline_r52.view import finish,encode

def main():
    out=fresh(ROOT/'bridge_viewer');r=read(ROOT/'controls/result.json');cases=read(ROOT/'main/cases.json')
    lookup={row['key']:row for row in r['hidden_bridges']};indices={u['unit_uid']:i for i,u in enumerate(r['units'])};z=np.load(ROOT/'controls/arrays.npz');J,parents=z['J'],z['parents'];rows=[]
    for c in cases:
        i=indices[c['unit_uid']];u=r['units'][i];side=u['pair']['common_hidden_seams'][0];methods=[];details={}
        for arm in ('A4','A5'):
            key='u%d_%s_%s'%(i,arm,side);record=lookup[key];details[arm]=record
            for name,label in [('truth','真实连续来源／虚拟切分'),('G1','实际 G1 补间')]:
                if key+'_'+name not in z:continue
                P=relative_fk(z[key+'_'+name],J,parents);d3=np.r_[np.zeros(3),np.sqrt(np.mean((np.diff(P,n=3,axis=0)*27000)**2,axis=(1,2)))];methods.append(dict(name=arm+' · '+label,data=encode(P),jerk=d3.tolist()))
        rows.append(dict(label=c['dataset']+' · '+c['source_uid']+' · '+side,frames=56,origin=0,audio='',methods=methods,
            meta=dict(scope='48帧隐藏动作的真实G1补间；同一缺口局部时钟。A4/A5原录制帧按移动边界不同，绝不声称同一绝对帧。缺失G1表示拒绝构造，不用来源替代。',
                      regions=[dict(kind='OBSERVED_CONTEXT',interval=[0,4]),dict(kind='HIDDEN_GAP_GENERATED_GEOMETRY',interval=[4,52]),dict(kind='OBSERVED_CONTEXT',interval=[52,56])],
                      details=details,unit_uid=c['unit_uid'],root_contact='UNKNOWN',semantic_completeness='UNKNOWN',neural_generation=False)))
    finish(out,rows,parents);page=(out/'index.html').read_text(encoding='utf8')
    page=page.replace('R52','R60').replace('同一时间轴上的真实效果','连续原动作与实际几何补间')
    page=page.replace('保留 Heuristic 强基线；生成比较同一完整来源与几何基线，修复评价时间坐标。来源阶段为观测运动学代理，世界 Root／接触及 Native24 尚未取得完整资格。','固定主预览来源的实际补间，不挑最好结果。每组独立对照真值与G1；A4/A5来源绝对帧不同，仅按缺口相对时间同步。没有神经生成或物理资格。')
    page=page.replace("+' · 音乐全局 '+(row.origin+frame)+' 帧'","+' · 缺口局部时钟（来源帧见详情）'")
    page=page.replace('(performance.now()-wall)*.03','(performance.now()-wall)*.03*+$(\'rate\').value')
    page=page.replace("$('rate').onchange=()=>{audio.defaultPlaybackRate", "$('rate').onchange=()=>{if(!row.audio&&running){base=frame;wall=performance.now()}audio.defaultPlaybackRate")
    (out/'index.html').write_text(page,encoding='utf8')
    write(out/'cases.json',[dict(label=row['label'],unit_uid=row['meta']['unit_uid'],panels=[m['name'] for m in row['methods']]) for row in rows])
    write(out/'protocol.json',dict(cases=len(rows),selection='SAME_PRE_OUTCOME_MAIN_CASES_FIRST_INTERVENED_SIDE_NO_NEW_SELECTION',arrays_sha256=sha(ROOT/'controls/arrays.npz'),neural_generation=False))
    (ROOT/'index.html').write_text('<meta charset="utf-8"><style>body{font:20px system-ui;max-width:1100px;margin:50px auto;color:#173e36;background:#f5f1e8}a{display:block;margin:24px}</style><h1>R60 · 阶段支持、原语时长与连接归因</h1><a href="main/viewer/index.html">① 连续来源／R47原语／移动切口：9组原速回放</a><a href="main/video/index.html">② 4段来源边界视频</a><a href="bridge_viewer/index.html">③ 原生连续动作／实际几何补间：相同预选来源</a><p>切分保护是运动学代理；没有新训练、神经重演、人工阶段真值或完整物理资格。实际结论见 reports/r60_release/report.md。</p>',encoding='utf8')
    print('R60_BRIDGE_VIEW',len(rows),flush=True)
if __name__=='__main__':main()
