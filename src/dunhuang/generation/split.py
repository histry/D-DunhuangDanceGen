"""Disjoint recording-level supervision splits; never split overlapping windows."""


def split_training_rows(windows, task, roles):
    groups = sorted({row[0] for row in windows})
    if task == 'paired':
        if any(g not in roles for g in groups):
            raise ValueError('Paired groups require explicit train/val roles')
        training = [row for row in windows if roles[row[0]] == 'train']
        validation = [row for row in windows if roles[row[0]] == 'val']
        return training, validation, sorted({row[0] for row in validation})
    if task not in ('motion', 'upperbody_probe'):
        raise ValueError('Unknown training task')
    if not groups:
        return [], [], None
    validation_group = groups[-1]
    return ([row for row in windows if row[0] != validation_group],
            [row for row in windows if row[0] == validation_group], validation_group)
