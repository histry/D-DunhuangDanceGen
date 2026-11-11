import torch
from dunhuang.generation.conditions import source_condition,target_supervision

def test_reference_and_supervision_are_independent_and_train_infer_match():
    ref=torch.randn(1,150,151);valid=torch.ones_like(ref,dtype=torch.bool)
    phase=torch.arange(150).float().reshape(1,150,1)/150
    train=source_condition(ref,valid,phase,'completion_upper8')
    inference=source_condition(ref.clone(),valid,phase,'completion_upper8',[0])
    assert torch.equal(train.value,inference.value) and torch.equal(train.known,inference.known)
    target=torch.randn_like(ref)
    mask=target_supervision(target,valid,train)
    assert mask[:,:,79:127].sum()==40*48 and not mask[:,:,:79].any()
    target.fill_(7)
    assert torch.equal(train.value,inference.value)  # Target cannot overwrite source condition.
    dense=source_condition(ref,valid,phase)
    assert target_supervision(target,valid,dense,'full_all').all()

def test_global_visibility_policy_preserves_overlap():
    global_ref=torch.randn(1,225,151)
    reference=torch.cat((global_ref[:,:150],global_ref[:,75:]),0)
    phase=torch.zeros(2,150,1)
    c=source_condition(reference,torch.ones_like(reference,dtype=torch.bool),phase,
        'completion_upper8',[0,75])
    assert torch.equal(c.value[0,75:],c.value[1,:75])
    assert torch.equal(c.known[0,75:],c.known[1,:75])
