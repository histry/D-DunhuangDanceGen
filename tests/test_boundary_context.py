import pytest
def test_boundary_context_zero_init_and_unknown_values_do_not_leak():
    torch=pytest.importorskip("torch")
    from dunhuang.research.boundary_context.adapter import BoundaryContextAdapter
    torch.manual_seed(12);model=BoundaryContextAdapter().eval()
    x=torch.randn(2,150,151);known=torch.zeros_like(x,dtype=torch.bool);known[:,52:60,4:]=True;known[:,90:98,4:]=True;music=torch.randn(2,150,512)
    assert torch.equal(model(x,known,music),torch.zeros(2,150,512))
    torch.nn.init.normal_(model.output.weight,std=.01)
    other=x.clone();other[~known]+=100
    assert torch.equal(model(x,known,music),model(other,known,music))
    known[:,:,:4]=True
    with pytest.raises(ValueError):model(x,known,music)

def test_masked_so3_boundary_loss_has_finite_gradient():
    torch=pytest.importorskip("torch");pytest.importorskip("pytorch3d")
    from dunhuang.research.boundary_context.train import loss
    x=torch.zeros(1,150,151);x[:,:,7:]=torch.tensor([1,0,0,0,1,0]*24)
    y=(x+torch.randn_like(x)*.001).requires_grad_()
    unknown=torch.zeros_like(x,dtype=torch.bool);unknown[:,60:90,4:]=True
    value=loss(y,x,unknown,60,90,(torch.ones(151),torch.zeros(151)));value.backward()
    assert torch.isfinite(value) and torch.isfinite(y.grad).all()
    assert y.grad[~unknown].abs().max()==0
