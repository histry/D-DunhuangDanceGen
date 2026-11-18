"""Post-hoc engineering selection, not a new trained model or physical certificate."""
import argparse,base64,gzip,json
from pathlib import Path
from dunhuang.research.complete_r36.core import read,write,sha
from .view import run as view

def decision(row):
    if row['certificates']['G0']['chart_eligible']:return dict(method='G0',status='KEEP_OLD_DOMAIN_ELIGIBLE_QUALITY_UNVERIFIED')
    if row['certificates']['G1']['status']=='GEOMETRICALLY_ADMITTED_NOT_PHYSICS' and row['screens'].get('G1',{}).get('numerical_screen',False):return dict(method='G1',status='G1_DOMAIN_AND_HISTORICAL_NUMERICAL_FILTER_NOT_PHYSICS')
    return dict(method=None,status='REJECT_CURRENT_ENDPOINT_PAIR')

def run(main,out):
    main=Path(main);out=Path(out)
    if out.exists():raise ValueError('Fresh engineering readout')
    meta=read(main/'result.json');rows=[dict(key=r['key'],**decision(r)) for r in meta['records']];write(out/'decision.json',dict(records=rows,input_arrays_sha256=sha(main/'arrays.npz'),policy='PRESERVE_DOMAIN_VALID_G0; FOR_OUT_OF_DOMAIN_TRY_REGISTERED_G1_AND_REGISTERED_NUMERICAL_FILTER_ELSE_REJECT',posthoc_engineering_policy=True,hidden_truth_read_by_decision=False,parameters_tuned_on_val=False,physical_qualified=False,can_route=False));view(main,out/'viewer')
    target=out/'viewer';text=(target/'data.js').read_text(encoding='utf8');encoded=text.split('"')[1];data=json.loads(gzip.decompress(base64.b64decode(encoded)));lookup={r['key']:r for r in rows}
    for case in data['cases']:
        selected=lookup[case['id']];case['engineering_decision']=selected;case['scope']=('本接法已拒绝，没有采用输出。' if selected['method'] is None else '研究性采用'+selected['method']+'，仍未取得物理资格。')+selected['status']+' · 事后工程选择，不是新模型或物理通过。'+case['scope']
        if selected['method'] is not None:
            p=next(p for p in case['panels'] if p['name']==selected['method']);case['panels'].append(dict(p,name='Selected · '+selected['method']))
    (target/'data.js').write_text('window.R42_B64="'+base64.b64encode(gzip.compress(json.dumps(data,ensure_ascii=False,separators=(',',':')).encode(),mtime=0)).decode()+'";',encoding='utf8');html=(target/'index.html').read_text(encoding='utf8').replace('相同观察端点：冻结五次桥与分段 SO(3) 桥','研究性选择：保留、采用候选或拒绝接法');html=html.replace('</header>','<p>本页为事后工程选择，未预登记为新的性能胜利。Selected复用已有真实数组；拒绝案例没有采用输出。旧桥域内也不代表自然或Native合格。</p></header>');(target/'index.html').write_text(html,encoding='utf8');js=(target/'viewer.js').read_text(encoding='utf8');old="selected=current.panels.map((_,i)=>current.panels.some(p=>p.name==='FrozenRTS')?[0,3,5].includes(i):i<4);";assert old in js;js=js.replace(old,"selected=current.panels.map(p=>p.reference||p.name==='G0'||p.name.startsWith('Selected')||(current.engineering_decision.method===null&&p.name==='G1'));");(target/'viewer.js').write_text(js,encoding='utf8');p=read(target/'protocol.json');p.update(engineering_decision_sha256=sha(out/'decision.json'),files=[dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(target.iterdir()) if f.name!='protocol.json']);write(target/'protocol.json',p)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.main,a.out)
