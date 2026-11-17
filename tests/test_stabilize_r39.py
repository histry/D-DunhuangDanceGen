import numpy as np
import pytest
from dunhuang.research.stabilize_r39.smoothing import constrained_rts,model,residuals

def test_constrained_rts_matches_dense_gaussian_equality_solution():
    n=12;dt=.1;obs=.07;jerk=3.;y=np.random.default_rng(3).normal(0,.1,(n,1));F,Q=model(dt,jerk);precision=np.zeros((3*n,3*n));rhs=np.zeros(3*n)
    for t in range(n-1):
        A=np.zeros((3,3*n));A[:,3*t:3*t+3]=-F;A[:,3*t+3:3*t+6]=np.eye(3);precision+=A.T@np.linalg.solve(Q,A)
    constraints=[]
    for t in (0,n-1):
        for j in range(3):c=np.zeros(3*n);c[3*t+j]=1;constraints.append(c)
    for t in (1,2,n-3,n-2):c=np.zeros(3*n);c[3*t]=1;constraints.append(c)
    for t in range(3,n-3):precision[3*t,3*t]+=1/obs**2;rhs[3*t]=y[t,0]/obs**2
    C=np.stack(constraints);K=np.block([[precision,C.T],[C,np.zeros((len(C),len(C)))]])
    exact=np.linalg.solve(K,np.r_[rhs,np.zeros(len(C))])[:3*n].reshape(n,3)[:,0]
    actual,info=constrained_rts(y,obs,jerk,dt=dt)
    assert np.allclose(actual[:,0],exact,atol=1e-8)
    assert np.array_equal(actual[:3],np.zeros((3,1))) and np.array_equal(actual[-3:],np.zeros((3,1)))

def test_root_projection_is_meter_bounded_and_preserves_protected_rows():
    import torch
    from dunhuang.research.stabilize_r39.projection import continuous_root
    from dunhuang.research.transition_r37.geometry import c2_envelope
    anchor=torch.zeros(2,256,3);proposal=torch.ones_like(anchor);scale=torch.tensor([.2,2.,.05]);w=torch.tensor(c2_envelope(256,[104,152]),dtype=torch.float32)
    out=continuous_root(proposal,anchor,w,scale,.15);physical=(out-anchor)/scale
    assert torch.all(physical.norm(dim=-1)<=.15*w+1e-7)
    assert torch.equal(out[:,:105],anchor[:,:105]) and torch.equal(out[:,151:],anchor[:,151:])
    assert physical[:,105].norm(dim=-1).max()<.001

def test_residual_solver_preserves_c4_and_outside_and_so3():
    from scipy.spatial.transform import Rotation
    from dunhuang.research.transition_r37.geometry import c2_envelope
    n=256;w=c2_envelope(n,[104,152]);base=np.zeros((n,139),np.float32);base[:,7:]=np.broadcast_to(np.eye(3)[:2],(n,22,2,3)).reshape(n,132)
    raw=base.copy();raw[:,0]=.3;raw[:,4]=.1*w*np.sin(np.arange(n)*2)
    rv=np.zeros((n,22,3));rv[:,:,0]=.2*w[:,None]*np.sin(np.arange(n)*2)[:,None];raw[:,7:]=Rotation.from_rotvec(rv.reshape(-1,3)).as_matrix().reshape(n,22,3,3)[...,:2,:].reshape(n,132)
    cal=dict(root_observation_sigma_m=.03,root_process_jerk_m_s3=2.,body_observation_sigma_rad=.05,body_process_jerk_rad_s3=5.,root_budget_m=.15)
    out,meta=residuals(raw,base,[104,152],cal,np.ones(3),np.zeros(3),.6)
    assert np.array_equal(out[:104],raw[:104]) and np.array_equal(out[152:],raw[152:]) and np.array_equal(out[:,:4],raw[:,:4])
    assert np.max(abs(np.diff(out[:,4],n=3)))<np.max(abs(np.diff(raw[:,4],n=3)))
    from dunhuang.research.transition_r37.evaluate import decode
    R,_=decode(out);assert np.max(abs(np.linalg.det(R)-1))<1e-6
    assert np.array_equal(out[104:107,4:],base[104:107,4:])

def test_invalid_frames_or_scales_rejected():
    with pytest.raises(ValueError):constrained_rts(np.zeros((4,3)),.01,1.)
    with pytest.raises(ValueError):constrained_rts(np.full((10,3),np.nan),.01,1.)
    with pytest.raises(ValueError):constrained_rts(np.zeros((10,3)),0.,1.)

def test_relative_root_derivatives_do_not_require_unknown_constant_j0():
    from dunhuang.research.stabilize_r39.root_contract import translation_channels
    t=np.arange(80)/30;root=np.c_[t,t*t,.2*np.sin(t)];a=translation_channels(root,30);b=translation_channels(root+[.03,-.41,.12],30)
    for k in a:assert np.allclose(a[k],b[k],atol=1e-10)

def test_offset_diagnostic_uses_equal_real_non_circular_support():
    from dunhuang.research.stabilize_r39.content_diagnostics import support,OFFSETS
    t,m=support(120);assert m.shape==(5,90) and t[0]==15 and t[-1]==104
    for k,d in enumerate(OFFSETS):assert np.array_equal(m[k],t+d) and m[k].min()>=0 and m[k].max()<120

def test_actual_diagnostic_renderer_emits_finite_curve_coordinates():
    import base64,gzip,json,shutil,subprocess
    from pathlib import Path
    from dunhuang.research.stabilize_r39.view import diagnostic_script
    node=shutil.which('node')
    if not node:pytest.skip('Node needed for actual JavaScript renderer regression')
    payload=dict(cases=[dict(id='fixture',kind='bridge',frames=5,audio=None,panels=[dict(title='fixture',track='t')],summary={},scope='fixture',diagnostics_variants=[{'Geometry':{'jerk':[0,1,3,1,0]}}])],tracks={'t':dict(shape=[5,22,3],data=base64.b64encode(np.zeros((5,22,3),'<f4').tobytes()).decode())},parents=[-1]+list(range(21)))
    encoded=base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode()
    js=r'''
const calls=[];const elements={};
function make(id=''){const context={fillRect(){},clearRect(){},beginPath(){},stroke(){},fill(){},arc(){},fillText(){},moveTo(x,y){if(id==='diagnostic')calls.push([x,y]);},lineTo(x,y){if(id==='diagnostic')calls.push([x,y]);}};return {id,width:1400,height:150,value:'',dataset:{},selectedOptions:[{textContent:'Jerk'}],pause(){},load(){},removeAttribute(){},append(){},replaceChildren(){},getContext(){return context;}};}
global.document={getElementById(id){return elements[id]||(elements[id]=make(id));},querySelectorAll(){return [];},createElement(){return make();}};
for(const [id,value] of Object.entries({seed:'0',metric:'jerk',view:'local',rate:'1'}))document.getElementById(id).value=value;
global.requestAnimationFrame=()=>{};global.window={R36_B64:ENCODED};process.on('unhandledRejection',e=>{console.error(e);process.exit(1);});
'''.replace('ENCODED',json.dumps(encoded))
    source=(Path(__file__).resolve().parents[1]/'src/dunhuang/research/transition_r37/web/viewer.js').read_text(encoding='utf8')
    js+=diagnostic_script(source)+r'''
setTimeout(()=>{if(calls.length<5||!calls.every(p=>p.every(Number.isFinite))||!calls.some(p=>p[1]<50))throw Error('Invalid diagnostic canvas path');console.log('FINITE_DIAGNOSTIC_PATH');},100);
'''
    result=subprocess.run([node,'-'],input=js,text=True,encoding='utf8',capture_output=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert 'FINITE_DIAGNOSTIC_PATH' in result.stdout

def test_content_controls_preserve_common_population_without_training_donors():
    from dunhuang.research.stabilize_r39.content_diagnostics import donor_population
    def row(uid,n,a,role='val'):return dict(unit_uid=uid,frames=n,audio_sha256=a,source_group=a,role=role)
    registered,eligible,donors,excluded=donor_population([row('a',120,'a'),row('b',120,'b'),row('long',360,'a'),row('train',360,'b','train')])
    assert registered==[0,1,2] and eligible==[0,1] and donors=={0:1,1:0}
    assert len(excluded)==1 and excluded[0]['unit_uid']=='long'
