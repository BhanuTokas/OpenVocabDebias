"""Error directions on synthetic embeddings with a planted bias."""

import numpy as np
import torch
import torch.nn.functional as F

from build_caption_vocab import noun_phrases, pick
from discover_biases import prompt_scores
from clip_debias.discovery import (
    as_prompt,
    direction_scores,
    cluster_error_directions,
    error_directions,
    name_by_cosine,
    remove_class_direction,
    load_vocabulary,
    residual_scores,
)


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


def test_direction_scores_keep_a_bias_shared_by_both_classes():
    v, y, err, b = planted()
    dirs, _ = error_directions(v, y, err)  # both classes point along b
    zb = (v @ b).numpy()
    corr = lambda a: abs(np.corrcoef(a.numpy(), zb)[0, 1])
    assert corr(residual_scores(v, None, dirs)[:, 1]) < 0.5  # a joint fit strips it
    z = direction_scores(v, None, dirs)
    assert corr(z[:, 0]) > 0.9 and corr(z[:, 1]) > 0.9


def test_load_vocabulary_is_a_union_without_repeats(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("water surface\n\nforest canopy\n")
    b.write_text("forest canopy\nsandy beach\n")
    assert load_vocabulary([str(a), str(b)]) == [
        "water surface",
        "forest canopy",
        "sandy beach",
    ]


def test_prompt_scores_do_not_reward_labelling_the_class():
    # concept agrees with the class on 95% of examples, as in Waterbirds
    y = np.array([0] * 95 + [0] * 5 + [1] * 5 + [1] * 95)
    c = np.array([0] * 95 + [1] * 5 + [0] * 5 + [1] * 95)
    s = prompt_scores(y.astype(bool), c.astype(bool).astype(int), y)
    assert s["prompt_acc"] == 0.95
    assert s["prompt_bal"] == 0.5 and s["prompt_worst"] == 0.0
    good = prompt_scores(c.astype(bool), c, y)
    assert good["prompt_bal"] == 1.0 and good["prompt_worst"] == 1.0


def fake_tagger(words):
    nouns = {"bird", "branch", "tree", "water", "lake", "woman", "dress"}
    adjs = {"small", "tall", "blue", "red"}
    return [(w, "NN" if w in nouns else "JJ" if w in adjs else "DT") for w in words]


def test_noun_phrases_from_a_caption():
    got = noun_phrases(
        "A small bird sitting on a tall tree branch near blue water", fake_tagger
    )
    assert {"small bird", "tall tree branch", "blue water"} <= got
    assert {"bird", "tree", "branch", "water"} <= got
    assert not any(p.endswith(("small", "tall", "blue")) for p in got)


def test_pick_takes_errors_and_as_many_correct_per_class():
    y = np.array([0] * 50 + [1] * 50)
    err = np.zeros(100, bool)
    err[:5], err[50:80] = True, True
    out = pick(y, err, per_class=10, rng=np.random.default_rng(0))
    for k, n in ((0, 5), (1, 10)):
        assert sum(1 for _, c, e in out if c == k and e) == n
        assert sum(1 for _, c, e in out if c == k and not e) == n
    assert all(err[i] == e for i, _, e in out)


def test_clusters_split_two_biases_within_a_class():
    g = torch.Generator().manual_seed(0)
    D, n = 32, 3000
    a, b = F.normalize(torch.randn(2, D, generator=g), dim=-1)
    v = torch.randn(n, D, generator=g) * 0.1
    y = np.zeros(n, int)
    err = np.zeros(n, bool)
    err[:150], err[150:300] = True, True
    v[:150] += 2 * a  # errors of one kind
    v[150:300] += 2 * b  # and of another
    dirs, classes, ids, sizes = cluster_error_directions(v, y, err, max_k=3)
    assert classes == [0, 0] and sorted(sizes) == [150, 150]
    cos = (dirs @ torch.stack([a, b]).T).abs()
    assert cos.max(dim=0).values.min() > 0.9
    one, *_ = cluster_error_directions(v, y, err, max_k=1)
    assert len(one) == 1


def test_remove_class_direction_is_orthogonal_to_the_class_difference():
    v, y, err, _ = planted()
    dirs, _ = error_directions(v, y, err)
    out = remove_class_direction(dirs, v, y)
    c = F.normalize(v[y == 1].mean(0) - v[y == 0].mean(0), dim=0)
    assert (out @ c).abs().max() < 1e-5
    assert torch.allclose(out.norm(dim=1), torch.ones(len(out)))


def test_name_by_cosine_ranks_by_text_similarity():
    text = torch.eye(3)
    pos, neg = name_by_cosine(
        ["a", "b", "c"], text, torch.tensor([0.0, 1, 0]), n=1, center=False
    )
    assert pos[0][0] == "b" and neg[0][0] in {"a", "c"}
