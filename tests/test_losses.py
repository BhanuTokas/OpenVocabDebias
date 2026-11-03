"""Losses on synthetic batches: values at known extremes, AMP-safe dtypes."""

import torch

from clip_debias.losses import cka_alignment_loss, decorrelation_loss, promotion_loss


def test_decorrelation_floor_for_independent_inputs():
    g = torch.Generator().manual_seed(0)
    B, d, r = 4096, 32, 3
    loss = decorrelation_loss(
        torch.randn(B, d, generator=g), torch.randn(B, r, generator=g)
    )
    assert loss < 3 * r / B  # floor is about r / B


def test_decorrelation_detects_a_concept_in_the_embedding():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(512, 1, generator=g)
    embed = torch.cat([z, torch.randn(512, 7, generator=g)], 1)  # feature 0 = concept
    assert decorrelation_loss(embed, z) > 0.1  # ~ 1 / d = 0.125


def test_decorrelation_conditional_on_labels_removes_label_driven_correlation():
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 2, (2048,), generator=g)
    # embed and score both depend on y only: correlated marginally, not within class
    embed = y[:, None].float() + 0.1 * torch.randn(2048, 4, generator=g)
    z = y[:, None].float() + 0.1 * torch.randn(2048, 1, generator=g)
    assert decorrelation_loss(embed, z) > 0.5
    assert decorrelation_loss(embed, z, labels=y) < 0.01


def test_decorrelation_is_fp32_under_half_inputs():
    g = torch.Generator().manual_seed(0)
    loss = decorrelation_loss(
        torch.randn(256, 16, generator=g).half(),
        torch.randn(256, 2, generator=g).half(),
    )
    assert loss.dtype == torch.float32 and torch.isfinite(loss)


def test_promotion_loss_zero_when_prediction_matches_scores():
    z = torch.randn(256, 2, generator=torch.Generator().manual_seed(0))
    assert promotion_loss(3 * z + 1, z) < 1e-4  # scale- and shift-free
    assert promotion_loss(-z, z) < 1e-4  # sign-free


def test_cka_alignment_zero_for_identical_inputs():
    x = torch.randn(128, 8, generator=torch.Generator().manual_seed(0))
    assert cka_alignment_loss(x, x).abs() < 1e-4
