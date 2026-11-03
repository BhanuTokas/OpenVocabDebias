"""Error directions on synthetic embeddings with a planted bias."""

import numpy as np
import torch
import torch.nn.functional as F

from clip_debias.discovery import as_prompt, error_directions, residual_scores


def planted(n=4000, D=32, seed=0):
    """Embeddings with a bias direction b; within each class the model errs on
    the examples whose bias score points the 'wrong' way."""
    g = torch.Generator().manual_seed(seed)
    b = F.normalize(torch.randn(D, generator=g), dim=0)
    y = np.random.default_rng(seed).integers(0, 2, n)
    s = torch.randn(n, generator=g)
    v = torch.randn(n, D, generator=g) * 0.3 + 2 * s[:, None] * b
    err = np.where(y == 0, s.numpy() > 1.0, s.numpy() < -1.0)
    return v, y, err, b


def test_error_directions_recover_the_planted_bias():
    v, y, err, b = planted()
    dirs, classes = error_directions(v, y, err)
    assert classes == [0, 1]
    assert torch.allclose(dirs.norm(dim=1), torch.ones(2))
    assert (dirs @ b).abs().min() > 0.9


def test_error_directions_skip_classes_with_too_few_errors():
    v, y, err, _ = planted()
    err = err & (y == 0)
    _, classes = error_directions(v, y, err)
    assert classes == [0]


def test_residual_scores_remove_the_prompted_concept():
    v, y, err, b = planted()
    dirs, _ = error_directions(v, y, err)
    z = residual_scores(v, b[None], dirs)  # the prompted subspace is b itself
    zb = v @ b
    for j in range(z.shape[1]):
        assert abs(np.corrcoef(z[:, j].numpy(), zb.numpy())[0, 1]) < 0.05
    z0 = residual_scores(v, None, dirs)  # nothing prompted: the bias stays
    assert abs(np.corrcoef(z0[:, 0].numpy(), zb.numpy())[0, 1]) > 0.9


def test_as_prompt():
    assert as_prompt("a forest") == "a forest"
    assert as_prompt("makeup") == "a photo of makeup"
