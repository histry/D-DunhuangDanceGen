import numpy as np
import pytest
from dunhuang.research.bridge_inpainting.audit import context_binding
from dunhuang.research.bridge_inpainting.study import smooth
CFG=dict(left_context=[52,60],right_context=[90,98],edit=[60,90],smooth_window=9,smooth_order=3)
def motion():
    x=np.zeros((200,151),np.float32);x[:,7:]=np.tile([1,0,0,0,1,0],24);return x
def test_context_contains_true_generated_root_and_all_rotations_without_clipping():
    x=motion();x[52:60,4]=3
    values,mask,outside=context_binding(x,np.ones(151),np.zeros(151),CFG)
    assert outside==8 and not mask[:,:,:4].any() and mask[:,:,4:7].sum()==16*3
    assert values[0,52,4]==3
def test_smoothing_control_protects_all_outside_edit_and_endpoints():
    x=motion();x[75,4]=.1;y=smooth(x,CFG)
    mask=np.ones(200,bool);mask[60:90]=False
    assert np.array_equal(x[mask],y[mask]) and np.array_equal(x[60:62],y[60:62]) and not np.array_equal(x[70:80,4],y[70:80,4])
def test_hard_inpaint_restores_context_on_the_final_reverse_step():
    torch=pytest.importorskip("torch")
    from dunhuang.sampling import constrained_ddim
    class D:
        betas=torch.zeros(1000);n_timestep=1000;guidance_weight=2;clip_denoised=True
        alphas_cumprod=torch.linspace(.999,.001,1000)
        def model_predictions(self,x,cond,t,**kwargs):return None,torch.ones_like(x)*.9
    x=torch.zeros(1,150,151);x[:,52:60,4:]=-.7;x[:,90:98,4:]=.3
    m=torch.zeros_like(x,dtype=torch.bool);m[:,52:60,4:]=True;m[:,90:98,4:]=True
    result,info=constrained_ddim(D(),torch.zeros(1,150,4800),x,m,guidance="hard")
    assert torch.equal(result[m],x[m]) and not info["post_generation_pose_paste"]
