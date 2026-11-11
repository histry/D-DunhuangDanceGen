import pytest
from dunhuang.generation.split import split_training_rows


def test_probe_and_motion_hold_out_whole_recordings():
    windows = [('b', 0), ('a', 0), ('a', 75), ('b', 75)]
    for task in ('upperbody_probe', 'motion'):
        train, val, group = split_training_rows(windows, task, {})
        assert group == 'b'
        assert set(x[0] for x in train) == {'a'}
        assert set(x[0] for x in val) == {'b'}


def test_paired_roles_are_explicit_and_separate():
    windows = [('b', 0), ('a', 0)]
    train, val, group = split_training_rows(windows, 'paired', {'a': 'val', 'b': 'train'})
    assert train == [('b', 0)] and val == [('a', 0)] and group == ['a']
    train, val, _ = split_training_rows(windows, 'paired', {'a': 'train', 'b': 'train'})
    assert train and not val
    with pytest.raises(ValueError):
        split_training_rows(windows, 'paired', {'a': 'train'})
