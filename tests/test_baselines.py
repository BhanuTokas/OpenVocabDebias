"""GroupDRO weights and the K-fold split used for cross-fitted discovery."""

import numpy as np
import pytest
import torch

from clip_debias.baselines import GroupDROLoss, fold_indices


def test_group_dro_upweights_the_worst_group_and_keeps_absent_ones():
    dro = GroupDROLoss(n_groups=3, eta=0.1)
    labels = torch.tensor([0, 0, 1, 1])
    groups = torch.tensor([0, 0, 1, 1])  # group 2 absent from the batch
    logits = torch.tensor([[5.0, 0], [5, 0], [0, 0.5], [0, 0.5]])  # group 1 is worse
    ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    for _ in range(5):
        q = dro.q.clone()
        dro(logits, labels, groups)
        assert torch.isclose(dro.q.sum(), torch.tensor(1.0))
        # no exp-update for the absent group: q2 / q0 only moves by group 0's step
        assert torch.isclose(
            dro.q[2] / dro.q[0], q[2] / (q[0] * torch.exp(0.1 * ce[:2].mean()))
        )
    assert dro.q[1] > dro.q[0]


def test_group_dro_loss_is_the_weighted_group_mean():
    dro = GroupDROLoss(n_groups=2, eta=0.0)  # eta 0: weights stay uniform
    logits = torch.tensor([[2.0, 0], [0, 2.0], [0, 0]])
    labels = torch.tensor([0, 0, 1])
    groups = torch.tensor([0, 0, 1])
    ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    expected = 0.5 * ce[:2].mean() + 0.5 * ce[2]
    assert torch.isclose(dro(logits, labels, groups), expected)


@pytest.mark.parametrize("n,k", [(10, 2), (4795, 5), (7, 3)])
def test_folds_partition_the_indices(n, k):
    folds = [fold_indices(n, k, f, seed=0) for f in range(k)]
    allidx = np.concatenate(folds)
    assert len(allidx) == n and len(np.unique(allidx)) == n
    assert max(len(f) for f in folds) - min(len(f) for f in folds) <= 1


def test_folds_depend_only_on_the_seed():
    assert np.array_equal(
        fold_indices(100, 5, 2, seed=1), fold_indices(100, 5, 2, seed=1)
    )
    assert not np.array_equal(
        fold_indices(100, 5, 2, seed=1), fold_indices(100, 5, 2, seed=2)
    )
