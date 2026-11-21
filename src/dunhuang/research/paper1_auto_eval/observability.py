"""Post-hoc readout of frozen controls: terminal orientation and exact edit stencils.

No perturbation, score definition or source is changed by this diagnostic.
It is not counted as a prospectively validated perceptual evaluator.
"""
import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .common import ROOT,read,write,prepare,sha
from .view import template
from dunhuang.research.complete_r36.core import relative_fk

def angular_acc(R):
    delta=R[:-1].swapaxes(-1,-2)@R[1:];w=Rotation.from_matrix(delta.reshape(-1,3,3)).as_rotvec().reshape(-1,22,3)*30
    return np.linalg.norm(((delta[:-1]@w[1:,...,None])[...,0]-w[:-1])*30,axis=-1)

def run(replay=False):
    root=ROOT/'controls';ctx=prepare();rows=read(root/'perturbations.json');z=np.load(root/'arrays.npz',allow_pickle=False);results=[];items=[]
    for key in sorted({r['case_key'] for r in rows}):
        own=[r for r in rows if r['case_key']==key];base=z[key+'__SourceOriginal__R'];Pb=relative_fk(base,ctx['J'],ctx['parents']);Ab=angular_acc(base);Db=np.linalg.norm(np.diff(Pb,n=3,axis=0)*30**3,axis=-1);variants=[];axes=[];angles=[]
        for r in own:
            R=z[key+'__'+r['label']+'__R'];P=relative_fk(R,ctx['J'],ctx['parents']);A=angular_acc(R);D=np.linalg.norm(np.diff(P,n=3,axis=0)*30**3,axis=-1);mask=np.zeros(len(R),bool);mask[r['write_frame_indices']]=True;stencil4=np.lib.stride_tricks.sliding_window_view(mask,4).any(1);stencil3=np.lib.stride_tricks.sliding_window_view(mask,3).any(1)
            scalar=lambda values,m:None if not m.any() else float(values[m].max())
            results.append(dict(case_key=key,source_uid=r['source_uid'],label=r['label'],local_FK_D3_max=scalar(D,stencil4),baseline_same_stencil_FK_D3_max=scalar(Db,stencil4),maximum_absolute_FK_D3_change=scalar(abs(D-Db),stencil4),local_SO3_acceleration_max=scalar(A,stencil3),baseline_same_stencil_SO3_acceleration_max=scalar(Ab,stencil3),maximum_absolute_SO3_acceleration_change=scalar(abs(A-Ab),stencil3),FK_position_max_abs_difference_m=float(np.max(abs(P-Pb))),SO3_rotation_matrix_max_abs_difference=float(np.max(abs(R-base))),edited_four_frame_D3_stencil_indices=np.flatnonzero(stencil4).tolist(),edited_three_frame_angular_stencil_indices=np.flatnonzero(stencil3).tolist(),scope='POST_HOC_FROZEN_WRITE_SUPPORT_DIAGNOSTIC_NOT_ARTISTIC_TRUTH'))
            W=np.empty_like(R);W[:,0]=R[:,0]
            for j in range(1,22):W[:,j]=W[:,ctx['parents'][j]]@R[:,j]
            endpoints=P[:,[20,21],None,:]+.18*W[:,[20,21]].swapaxes(-1,-2)
            residual=Rotation.from_matrix((base[:,[20,21]].swapaxes(-1,-2)@R[:,[20,21]]).reshape(-1,3,3)).as_rotvec().reshape(-1,2,3)
            angles.append((np.linalg.norm(residual,axis=-1).max(-1)*180/np.pi).tolist());axes.append(np.round(endpoints,5).tolist());variants.append(dict(label=r['label'],P=np.round(P,4).tolist()))
        items.append(dict(label=own[0]['source_uid'],classification='PERTURBATION',frames=len(base),core=len(base),status='腕部中心可能不动，但末端方向确实改变；三色轴线仅为0.18m统一方向示意',variants=variants,axes_by_variant=axes,wrist_error_degrees=angles,edit_by_variant=[r['write_frame_indices'] for r in own],evidence='此页从同一冻结旋转数组推导方向轴；没有放大旋转剂量。补充局部差分读域是事后诊断，不替代主协议或人类偏好。'))
    z.close();dest=ROOT/('observability_replay' if replay else 'observability');dest.mkdir(exist_ok=False);write(dest/'edit_support.json',results)
    t=template();assert 'for(let v of views){' in t;t=t.replace('for(let v of views){','for(let [vi,v] of views.entries()){');t=t.replace('p.onchange=()=>{',"p.value='2';p.onchange=()=>{")
    mark="c.stroke()}c.fillStyle='#276b89';";assert mark in t
    t=t.replace(mark,"c.stroke()}let ax=D.items[current].axes_by_variant?.[vi]?.[f];if(ax){for(let k=0;k<2;k++)for(let j=0;j<3;j++){let e=ax[k][j],px=150+115*(e[0]+.35*e[2]),py=180-115*(e[1]-.15*e[2]);c.strokeStyle=['#ed7d79','#8edb89','#82b8f0'][j];c.beginPath();c.moveTo(...p[20+k]);c.lineTo(px,py);c.stroke()}}c.fillStyle='#276b89';")
    # Display the recorded wrist residual, without changing the underlying poses.
    t=t.replace("$('evidenceText').textContent=x.evidence||", "$('evidenceText').textContent=(x.wrist_error_degrees?'当前腕部相对来源旋转差 '+x.wrist_error_degrees[k][Math.min(frame,x.frames-1)].toFixed(3)+'°； ':'')+(x.evidence||'')||")
    payload=dict(items=items,parents=ctx['parents'].tolist(),anonymous=False,task='naturalness',variant_select=True,intro='原始扰动未更换：腕部局部旋转不会改变身体22关节中的末端中心。这里显示实际旋转推导的腕部方向轴与SO(3)差值；同时保留原关节中心视图。')
    page=t.replace('__TITLE__','末端方向与冻结写域诊断').replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')));(dest/'orientation.html').write_bytes(page.encode())
    if replay:
        original=ROOT/'observability';diff=[n for n in ['edit_support.json','orientation.html'] if sha(dest/n)!=sha(original/n)];assert not diff,diff;write(ROOT/'receipts/observability_replay.json',dict(status='PASS',differences=diff,scope='POST_HOC_DERIVED_SCALARS_AND_VIEW_ONLY'))
    else:
        out=root/'viewer';(out/'orientation.html').write_bytes((dest/'orientation.html').read_bytes());p=out/'protocol.json';m=read(p);m['files']['orientation.html']=dict(bytes=(out/'orientation.html').stat().st_size,sha256=sha(out/'orientation.html'));m['orientation_diagnostic']='POST_HOC_READONLY_NO_SCIENTIFIC_PRODUCER_CHANGE';write(p,m)
        p=ROOT/'index.html';s=p.read_text(encoding='utf8');s=s.replace('</ol>','<li><a href="controls/viewer/orientation.html">腕部方向与完整编辑读域诊断（事后只读）</a></li></ol>');p.write_bytes(s.encode())
    print('AUTO_EVAL_ORIENTATION_DIAGNOSTIC',len(results),'REPLAY',replay,flush=True)
if __name__=='__main__':run('--replay' in sys.argv)
