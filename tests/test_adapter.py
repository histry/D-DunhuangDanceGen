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
    assert not mask[:, 30:35].any()
    assert not mask[:, 100:110].any()
    assert mask[:, 55:80, 79:127].all()

def test_trained_residual_ignores_hidden_storage_and_uses_phase():
    torch.manual_seed(7)
    adapter=PrimitiveAdapter().eval()
    with torch.no_grad():adapter.output.weight.normal_(0,.02)
    value=torch.randn(1,150,151);known=torch.zeros_like(value,dtype=torch.bool)
    known[:,:20,79:127]=True
    phase=torch.arange(150).float().reshape(1,150,1)/150
    altered=value.clone();altered[~known]=float('nan')
    with torch.no_grad():
        original=adapter(value,known,phase)
        torch.testing.assert_close(original,adapter(altered,known,phase))
        assert not torch.equal(original,adapter(value,known,phase*.5))

def test_unqualified_probe_requires_explicit_research_true():
    import pytest
    from dunhuang.generation.adapter import authorize_adapter
    payload=dict(base_sha256='base',condition_schema='soft_event_reference_v1',
        music_qualified=False,full_motion_qualified=False)
    for flag in (False,None,1,'true'):
        with pytest.raises(ValueError,match='qualification'):
            authorize_adapter(payload,'base',flag)
    assert authorize_adapter(payload,'base',True)['research_unqualified']
    with pytest.raises(ValueError,match='base'):
        authorize_adapter(payload,'different',True)
