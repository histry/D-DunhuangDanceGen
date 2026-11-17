import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.research.paired_event_r41.events import profiles,segment,clock_relations,validate_roles

CFG=dict(boundary_smoothing_sigma_frames=1.5,minimum_core_frames=90,maximum_core_frames=360,preferred_core_frames=180,boundary_grid_frames=6)

def test_long_hold_is_not_split_for_a_length_cap():
    p={k:np.zeros(700) for k in ('global','upper','lower','trunk')}
    spans,evidence=segment(p,[dict(type='pose_hold',evidence_span=[0,700])],CFG)
    assert spans==[[0,700]]
    assert evidence['protected_spans']==[[0,700]]

def test_partition_covers_all_real_frames_and_protects_turns():
    v=np.ones(720);v[150:175]=0;v[340:365]=0;v[530:555]=0;p={k:v for k in ('global','upper','lower','trunk')}
    result,_=segment(p,[dict(type='root_turn_span',evidence_span=[160,330])],CFG)
    assert result[0][0]==0 and result[-1][1]==720
    assert all(a[1]==b[0] for a,b in zip(result,result[1:]))
    assert all(not 160<a<330 for a,b in result)

def test_whole_record_derivative_is_not_restarted_at_event_boundaries():
    rv=np.zeros((200,22,3));rv[:,:,2]=np.arange(200)[:,None]**2*.0001
    R=Rotation.from_rotvec(rv.reshape(-1,3)).as_matrix().reshape(200,22,3,3)
    full=profiles(R)['omega'];local=profiles(R[90:150])['omega']
    assert not np.allclose(full[90],local[0])
    np.testing.assert_allclose(full[91:150],local[1:],atol=1e-9)

def row(start,end,role='train',raw='r',audio='a',group='g'):
    return dict(audio_frame_interval=[start,end],role=role,raw_motion_sha256=raw,audio_sha256=audio,source_group=group)

def test_clock_overlap_unknown_and_hard_negative_are_distinct():
    r=[row(0,120),row(10,130,raw='s'),row(80,200,raw='t'),row(140,260,raw='u')]
    p,n=clock_relations(r)
    assert p[0,1] and not n[0,1]
    assert not p[0,2] and not n[0,2]
    assert n[0,3] and not p[0,3]

def test_source_audio_and_music_group_cannot_cross_roles():
    for key in ('raw_motion_sha256','audio_sha256','source_group'):
        a=row(0,120);b=row(150,270,role='val',raw='new',audio='other',group='different');b[key]=a[key]
        with pytest.raises(ValueError,match='crosses roles'):validate_roles([a,b])
    with pytest.raises(ValueError,match='Test tensors'):validate_roles([row(0,120,role='test')])

def test_local_content_is_translation_equivariant_and_clock_channels_are_ignored():
    import torch
    from dunhuang.research.paired_event_r41.model import EventContent,RADIUS
    torch.manual_seed(1);torch.set_num_threads(1);m=EventContent().eval();a=torch.randn(1,170,50);d=torch.randn(1,170,274);mask=torch.ones_like(d,dtype=torch.bool);valid=torch.ones(1,170,dtype=torch.bool)
    with torch.no_grad():
        x,y,*_=m.content(a,d,mask,valid);u,v,*_=m.content(a[:,25:125],d[:,25:125],mask[:,25:125],valid[:,25:125])
        torch.testing.assert_close(x[:,50-RADIUS:90-RADIUS],u[:,25-RADIUS:65-RADIUS],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(y[:,50-RADIUS:90-RADIUS],v[:,25-RADIUS:65-RADIUS],rtol=1e-5,atol=1e-6)
        d[:,:,-2:]=123456;_,other,*_=m.content(a,d,mask,valid);torch.testing.assert_close(y,other)

def test_zero_content_cannot_create_a_local_absolute_index_pattern():
    import torch
    from dunhuang.research.paired_event_r41.model import EventContent,scores_for_offsets
    torch.manual_seed(2);model=EventContent().eval();d=torch.zeros(3,90,274);a=torch.zeros(3,90,50)
    with torch.no_grad():
        m,x,*_=model.content(a,d,torch.ones_like(d,dtype=torch.bool),torch.ones(3,90,dtype=torch.bool));scores=scores_for_offsets(m,x,[-15,-8,0,8,15])
    assert torch.max(abs(scores-scores[:,:1]))<1e-6

def test_shared_signal_offset_sign_and_actual_gradient_path():
    import torch
    from dunhuang.research.paired_event_r41.model import EventContent,scores_for_offsets,local_objective
    torch.manual_seed(3);master=torch.nn.functional.normalize(torch.randn(200,96),dim=-1);offsets=[-15,-8,0,8,15]
    for delta in offsets:
        m=master[21+delta:99+delta][None];d=master[21:99][None];s=scores_for_offsets(m,d,offsets);assert offsets[int(s.argmax())]==delta
    model=EventContent();x=torch.randn(2,90,274);music=torch.randn(2,90,50);mt,dt,*_=model.content(music,x,torch.ones_like(x,dtype=torch.bool),torch.ones(2,90,dtype=torch.bool));a,b,_=local_objective(mt,dt,[[3,9],[14,2]]);(a+b).backward()
    assert model.music[0].weight.grad.norm()>0 and model.motion[0].weight.grad.norm()>0
    assert model.local_music[0].weight.grad.norm()>0 and model.local_motion[0].weight.grad.norm()>0

def test_full_reconstruction_support_is_not_replaced_by_crops():
    import torch
    from dunhuang.research.paired_event_r41.model import EventContent
    from dunhuang.research.paired_event_r41.train import objective
    torch.manual_seed(4)
    class Pop:
        rows=[dict(frames=150) for i in range(4)]
        m=torch.randn(4,150,50);d=torch.randn(4,150,274)
        def batch(self,ids,origins=None):
            if origins is None:a=self.m[ids];b=self.d[ids]
            else:a=torch.stack([self.m[i,int(o[0])+1:int(o[0])+91] for i,o in zip(ids,origins)]);b=torch.stack([self.d[i,int(o[1])+1:int(o[1])+91] for i,o in zip(ids,origins)])
            return a,b,torch.ones_like(b,dtype=torch.bool),torch.ones(a.shape[:2],dtype=torch.bool)
        def relations(self,ids):return np.eye(len(ids),dtype=bool),~np.eye(len(ids),dtype=bool)
    cfg=dict(seed=41041,loss_weights=dict(global_=1.,clock=.1,local=.1,reconstruction=.1,equivariance=.05));cfg['loss_weights']['global']=cfg['loss_weights'].pop('global_');value,parts,meta=objective(EventContent(),Pop(),[0,1,2,3],cfg,0)
    assert meta['full_reconstruction_frames']==600 and meta['reconstruction_observed_values']==600*198
    assert torch.isfinite(value)

def test_clock_metrics_separate_index_prior_from_content_recovery():
    from dunhuang.research.paired_event_r41.evaluate import metrics
    offsets=[-15,-8,0,8,15];rows=[dict(source_group='a'),dict(source_group='b')];correct=np.broadcast_to(np.eye(5),(2,5,5)).copy();r=metrics(correct,rows,offsets)
    assert r['offset_accuracy']==1 and r['offset_mae_frames']==0 and r['equivariance_error_frames']==0
    prior=np.zeros_like(correct);prior[:,:,2]=1;r=metrics(prior,rows,offsets)
    assert r['offset_accuracy']==pytest.approx(.2) and r['equivariance_error_frames']>0
    r=metrics(np.ones_like(correct),rows,offsets);assert r['offset_accuracy']==pytest.approx(.2)

def test_failed_learnability_gate_cannot_construct_main_model_or_optimizer(tmp_path,monkeypatch):
    import json,sys
    from dunhuang.research.paired_event_r41 import train
    data=tmp_path/'data';data.mkdir();(data/'manifest.json').write_text('{}');checks=tmp_path/'checks';checks.mkdir();(checks/'result.json').write_text(json.dumps(dict(admitted_main=False)));out=tmp_path/'main'
    monkeypatch.setattr(train,'Population',lambda path:object())
    def forbidden(*a,**kw):raise AssertionError('A failed gate attempted training')
    monkeypatch.setattr(train,'initial',forbidden);monkeypatch.setattr(train.torch.optim,'AdamW',forbidden)
    monkeypatch.setattr(sys,'argv',['train','main','--data',str(data),'--checks',str(checks),'--out',str(out)]);train.main();r=json.loads((out/'result.json').read_text());assert not r['admitted'] and r['optimizer_steps']==0

def test_budget_regions_and_chain_diagnostics_use_actual_frame_support():
    from dunhuang.research.paired_event_r41.budget import summarize
    raw=np.zeros((1,256,139),np.float32);raw[:,:,7:]=np.tile(np.eye(3)[:2].reshape(6),22);J=np.zeros((22,3));J[:,1]=np.arange(22)*.01;parents=np.array([-1]+[0]*21)
    r=summarize(raw,raw.copy(),raw.copy(),J,parents)[0]
    assert r['zero_budget']['frames']==2 and r['ramp']['frames']==22 and r['central']['frames']==24
    assert all(r[k]['outside_fraction']==0 and r[k]['fk_rmse_m']==0 for k in ('zero_budget','ramp','central'))
