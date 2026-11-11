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


def test_phase_contract_and_restricted_reference_mask():
    import pytest
    from dunhuang.generation.adapter import masked_reference
    adapter = PrimitiveAdapter().eval()
    x = torch.zeros(1, 150, 151)
    known = torch.zeros_like(x, dtype=torch.bool)
    phase = torch.linspace(0, 1, 150).reshape(1, 150, 1)
    assert torch.equal(adapter(x, known, phase), torch.zeros(1, 150, 512))
    with pytest.raises(ValueError, match='phase'):
        adapter(x, known, torch.ones(1, 150, 2))
    value, mask = masked_reference(x, scope='upperbody_probe')
    assert mask[:, :, :79].sum() == 0 and mask[:, :, 127:].sum() == 0
    assert mask[:, 0:8, 79:127].all()
    assert not mask[:, 50:100].any()
