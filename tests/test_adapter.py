import torch
from dunhuang.generation.adapter import PrimitiveAdapter,masked_reference
from dunhuang.generation.inpainting import boundary_context

def test_reference_mask_prevents_hidden_target_leakage():
    target=torch.randn(2,150,151)
    value,mask=masked_reference(target)
    assert not mask[:,60:90].any()
    assert not value[~mask].any()
    adapter=PrimitiveAdapter().eval()
    assert torch.equal(adapter(value,mask),torch.zeros(2,150,512))
    modified=value.clone();modified[~mask]=1000
    # Even nonzero stale storage in hidden positions cannot enter the network.
    torch.testing.assert_close(adapter(value,mask),adapter(modified,mask))

def test_invalid_future_context_remains_unknown():
    x=torch.randn(1,150,151);valid=torch.ones_like(x,dtype=torch.bool);valid[:,-10:]=False
    value,mask=boundary_context(x,valid)
    assert not mask[:,-10:].any() and not value[:,-10:].any()
