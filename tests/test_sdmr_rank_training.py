import numpy as np
from dunhuang.research.sdmr_event_rank.train_rank import schedule,shuffled_labels

def test_fixed_train_only_schedule_and_all_lengths():
    tasks=[dict(role=role,pool='energy_matched',frames=L,source_group=str(g)) for role in ('train','val') for L in (90,150,240) for g in range(6)]
    meta=dict(config=dict(seed=43043,lengths=[90,150,240]),tasks=tasks)
    a=schedule(meta);assert a==schedule(meta) and len(a)==240
    for step,ids in enumerate(a):
        assert len(ids)==4 and len({tasks[i]['source_group'] for i in ids})==4
        assert all(tasks[i]['role']=='train' and tasks[i]['frames']==[90,150,240][step%3] for i in ids)

def test_shuffled_rank_is_repeatable_count_preserving():
    p=[True,False,False,True,False,False]
    assert np.array_equal(shuffled_labels(p,43),shuffled_labels(p,43))
    assert all(shuffled_labels(p,i).sum()==2 for i in range(10))
    assert any(not np.array_equal(p,shuffled_labels(p,i)) for i in range(10))
