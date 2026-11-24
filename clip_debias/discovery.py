"""
discovery.py — Error-driven discovery of biases the user did not prompt.

Given a trained classifier's mistakes, the within-class difference between the
mean CLIP image embedding of misclassified and correctly classified images
points at what the mistakes have in common.  Directions are made orthogonal to
the already-prompted concept subspace with one joint whitened fit, so only
residual biases remain, and are then named by the vocabulary phrases whose
zero-shot image scores correlate most with them.  The phrases at the two ends
of a direction form a proposed (pos, neg) prompt pair; a user confirms or edits
it (discover_biases.py writes them to an editable JSON file) and training uses
the confirmed pairs like any hand-written prompts (--extra_prompts).

The direction itself is not used for training: error directions also contain
the target (misclassified images look like the other class), and whenever the
class name is entangled with the bias in CLIP space (waterbird / water) no
direction-level filter separates the two.  Text prompts give far better
pseudo-labels in the rare groups.
"""

from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .clip_oracle import CLIPOracle, fit_leace_eraser, whitened_concept_scores

# Generic candidate phrases: scenes, people and appearance, image properties.
# Spurious features are mostly context or appearance, not other objects, and an
# object vocabulary (e.g. ImageNet classes) brings in target words such as bird
# species.  Pass objects=True to vocabulary() to add the ImageNet classes anyway.
SCENES = [
    "the ocean",
    "the sea",
    "a beach",
    "a lake",
    "a river",
    "a pond",
    "a waterfall",
    "a swamp",
    "a marsh",
    "a harbor",
    "a shore",
    "water",
    "a forest",
    "a bamboo forest",
    "a jungle",
    "the woods",
    "a park",
    "a garden",
    "a field",
    "a meadow",
    "a farm",
    "a mountain",
    "a hill",
    "a valley",
    "a canyon",
    "a desert",
    "a cave",
    "a glacier",
    "snow",
    "ice",
    "the sky",
    "clouds",
    "a sunset",
    "night",
    "a city street",
    "a highway",
    "a parking lot",
    "a bridge",
    "a building",
    "a skyscraper",
    "a house",
    "a kitchen",
    "a living room",
    "a bedroom",
    "a bathroom",
    "an office",
    "a classroom",
    "a library",
    "a store",
    "a restaurant",
    "a stadium",
    "a stage",
    "a photo studio",
    "an airport",
    "a train station",
    "a playground",
    "rocks",
    "sand",
    "grass",
    "tree branches",
    "flowers",
    "bushes",
    "a fence",
    "a wall",
    "a road",
    "dirt",
    "mud",
    "rain",
    "fog",
    "indoors",
    "outdoors",
]
PEOPLE = [
    "a man",
    "a woman",
    "a boy",
    "a girl",
    "a child",
    "a baby",
    "a teenager",
    "a young person",
    "a young adult",
    "a middle-aged person",
    "an old person",
    "an elderly person",
    "an old man",
    "an old woman",
    "blond hair",
    "dark hair",
    "brown hair",
    "red hair",
    "black hair",
    "gray hair",
    "white hair",
    "curly hair",
    "straight hair",
    "long hair",
    "short hair",
    "a bald head",
    "a beard",
    "a mustache",
    "makeup",
    "lipstick",
    "earrings",
    "a necklace",
    "a hat",
    "eyeglasses",
    "sunglasses",
    "a smile",
    "an open mouth",
    "a serious face",
    "pale skin",
    "dark skin",
    "tanned skin",
    "wrinkles",
    "a suit",
    "a tie",
    "a dress",
    "a uniform",
    "a celebrity",
]
IMAGE = [
    "a blurry photo",
    "a black and white photo",
    "a close-up photo",
    "a portrait",
    "studio lighting",
    "a dark photo",
    "a bright photo",
    "an overexposed photo",
    "a low resolution photo",
    "a painting",
    "a drawing",
    "a cartoon",
    "a photo with text",
    "a photo with a border",
]


def vocabulary(objects: bool = False) -> List[str]:
    vocab = SCENES + PEOPLE + IMAGE
    if objects:
        from torchvision.models import ResNet50_Weights

        vocab = vocab + ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    return list(dict.fromkeys(vocab))


def load_vocabulary(paths: List[str]) -> List[str]:
    """Union of one-phrase-per-line files, in order, without repeats."""
    phrases: List[str] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            phrases += [line.strip() for line in f if line.strip()]
    return list(dict.fromkeys(phrases))


def as_prompt(phrase: str) -> str:
    return (
        phrase if phrase.startswith(("a ", "an ", "the ")) else f"a photo of {phrase}"
    )


def error_directions(
    v: torch.Tensor, y: np.ndarray, err: np.ndarray, min_errors: int = 5
) -> Tuple[torch.Tensor, List[int]]:
    """(m, D) unit within-class directions mean(errors) − mean(correct), one per
    class with at least `min_errors` errors, and the classes they belong to."""
    dirs, classes = [], []
    for k in np.unique(y):
        m = y == k
        if (m & err).sum() < min_errors or (m & ~err).sum() < min_errors:
            continue
        dirs.append(v[m & err].mean(0) - v[m & ~err].mean(0))
        classes.append(int(k))
    if not dirs:
        raise SystemExit(f"no class has {min_errors} errors and {min_errors} correct")
    return F.normalize(torch.stack(dirs).float(), dim=-1), classes


def cluster_error_directions(
    v: torch.Tensor,
    y: np.ndarray,
    err: np.ndarray,
    max_k: int = 1,
    min_size: int = 10,
    min_errors: int = 5,
    seed: int = 0,
) -> Tuple[torch.Tensor, List[int], List[int], List[int]]:
    """Several error directions per class: the class's errors are split by
    k-means on their (unit) CLIP embeddings, k in 2..max_k picked by silhouette
    score among splits whose clusters all have min_size errors (k = 1 when none
    qualifies), and each cluster gives mean(cluster) − mean(correct).  Returns
    unit directions (m, D) and, per direction, its class, cluster id and size.
    max_k = 1 is error_directions."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    dirs, classes, ids, sizes = [], [], [], []
    for k in np.unique(y):
        m = y == k
        e, c = v[m & err].float(), v[m & ~err].float()
        if len(e) < min_errors or len(c) < min_errors:
            continue
        labels = np.zeros(len(e), int)
        x = F.normalize(e, dim=-1).numpy()
        best = -1.0
        for n in range(2, max_k + 1):
            if len(e) < n * min_size:
                break
            lab = KMeans(n, n_init=10, random_state=seed).fit_predict(x)
            if np.bincount(lab).min() < min_size:
                continue
            s = silhouette_score(x, lab)
            if s > best:
                best, labels = s, lab
        mu_c = c.mean(0)
        for j in range(labels.max() + 1):
            dirs.append(e[labels == j].mean(0) - mu_c)
            classes.append(int(k))
            ids.append(j)
            sizes.append(int((labels == j).sum()))
    if not dirs:
        raise SystemExit(f"no class has {min_errors} errors and {min_errors} correct")
    return F.normalize(torch.stack(dirs), dim=-1), classes, ids, sizes


def remove_class_direction(
    dirs: torch.Tensor, v: torch.Tensor, y: np.ndarray
) -> torch.Tensor:
    """Directions with the between-class mean difference projected out.  An
    error direction also points toward the other class, since misclassified
    images look like it; this keeps only what is not the class itself."""
    c = v[y == 1].float().mean(0) - v[y == 0].float().mean(0)
    c = F.normalize(c, dim=0)
    return F.normalize(dirs - (dirs @ c)[:, None] * c, dim=-1)


def residual_scores(
    v: torch.Tensor, prompted: torch.Tensor | None, dirs: torch.Tensor
) -> torch.Tensor:
    """(N, m) whitened scores of `dirs` after removing everything the prompted
    subspace (r, D) explains: one joint whitened fit with the prompted rows
    first, keeping the trailing block."""
    r = 0 if prompted is None else prompted.shape[0]
    basis = dirs if prompted is None else torch.cat([prompted.float().cpu(), dirs])
    return whitened_concept_scores(v, fit_leace_eraser(v, basis))[:, r:]


def direction_scores(
    v: torch.Tensor, prompted: torch.Tensor | None, dirs: torch.Tensor
) -> torch.Tensor:
    """(N, m) residual_scores of each direction on its own.  One joint fit over
    `dirs` would score a later class's direction only on what earlier classes do
    not explain, and the classes' directions are usually the same bias."""
    return torch.cat([residual_scores(v, prompted, d[None]) for d in dirs], dim=1)


@torch.no_grad()
def name_scores(
    oracle: CLIPOracle,
    vocab: List[str],
    v: torch.Tensor,
    z: torch.Tensor,
    n: int = 5,
    text: torch.Tensor | None = None,
    center: bool = True,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Phrases whose zero-shot image scores correlate most positively and most
    negatively with the score z (N,).  The text embeddings are mean-centred over
    the vocabulary first unless center=False; `text` reuses an encoding."""
    t = text if text is not None else encode_vocab(oracle, vocab)
    s = v.float() @ (t - t.mean(0) if center else t).T
    s = (s - s.mean(0)) / (s.std(0) + 1e-8)
    zz = (z.float() - z.float().mean()) / (z.float().std() + 1e-8)
    c = (s * zz[:, None]).mean(0).numpy()
    order = np.argsort(-c)
    pos = [(vocab[i], float(c[i])) for i in order[:n]]
    neg = [(vocab[i], float(c[i])) for i in order[::-1][:n]]
    return pos, neg


def encode_vocab(oracle: CLIPOracle, vocab: List[str]) -> torch.Tensor:
    return oracle.encode_text([as_prompt(p) for p in vocab]).float().cpu()


def name_by_cosine(
    vocab: List[str],
    text: torch.Tensor,
    d: torch.Tensor,
    n: int = 5,
    center: bool = True,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Retrieval naming (as in DrawTheName): phrases whose text embedding has
    the highest and lowest cosine with the direction d, after mean-centring the
    vocabulary's embeddings unless center=False."""
    t = F.normalize(text - text.mean(0) if center else text, dim=-1)
    c = (t @ F.normalize(d.float(), dim=0)).numpy()
    order = np.argsort(-c)
    pos = [(vocab[i], float(c[i])) for i in order[:n]]
    neg = [(vocab[i], float(c[i])) for i in order[::-1][:n]]
    return pos, neg


def load_prompt_file(path: str) -> List[Tuple[List[str], List[str]]]:
    """Accepted (pos, neg) prompt pairs from a discover_biases.py proposals file
    (entries with "accept": true)."""
    with open(path) as f:
        entries = json.load(f)["proposals"]
    pairs = [(list(e["pos"]), list(e["neg"])) for e in entries if e.get("accept")]
    if not pairs:
        raise ValueError(f"no accepted prompt pairs in {path}")
    return pairs


# ── Core concepts ─────────────────────────────────────────────────────────────
# The mirror image of bias discovery: within a (class × pseudo-group) cell the
# prompted bias is held fixed, so the difference between the mean CLIP image
# embedding of correctly and wrongly classified images points at cues the model
# under-uses.  Unlike biases, these are kept as image-space directions: words
# for object parts carry their habitat in CLIP text space ("wading" → water),
# so prompts re-introduce the bias.  Phrases only name a direction afterwards.

# Object parts and appearance, for naming core directions (birds: Waterbirds).
BIRD_PARTS = [
    "webbed feet",
    "long thin legs",
    "short legs",
    "perching feet",
    "talons",
    "a long thin beak",
    "a short conical beak",
    "a hooked beak",
    "a flat wide bill",
    "a long neck",
    "a short neck",
    "a plump body",
    "a slender body",
    "long pointed wings",
    "short rounded wings",
    "a long tail",
    "a short tail",
    "a crest",
    "colorful plumage",
    "dull brown feathers",
    "white feathers",
    "black feathers",
    "gray feathers",
    "yellow feathers",
    "red feathers",
    "blue feathers",
    "spotted feathers",
    "striped feathers",
    "a small bird",
    "a large bird",
    "a songbird",
    "a seabird",
    "a wading bird",
    "a diving bird",
    "a swimming bird",
    "a flying bird",
    "a bird perched on a branch",
    "a bird floating on water",
    "a bird standing",
    "a bird in flight",
]
CORE_VOCAB = {"birds": BIRD_PARTS, "people": PEOPLE}


def core_directions(
    v: torch.Tensor,
    y: np.ndarray,
    groups: np.ndarray,
    err: np.ndarray,
    min_errors: int = 20,
) -> Tuple[torch.Tensor, List[str]]:
    """Unit (D,) core directions: per (class × pseudo-group) cell with at least
    `min_errors` errors and as many correct images, mean(correct) − mean(errors),
    signed toward class 1 (binary targets); row 0 is their average (pooled), the
    rest are the cells.  Returns (directions (1 + cells, D), tags)."""
    dirs, tags = [], []
    for k in np.unique(y):
        for g in np.unique(groups):
            m = (y == k) & (groups == g)
            ne, nc = int((m & err).sum()), int((m & ~err).sum())
            if ne < min_errors or nc < min_errors:
                continue
            sign = 1.0 if k == 1 else -1.0
            dirs.append(sign * (v[m & ~err].mean(0) - v[m & err].mean(0)))
            tags.append(
                f"class {int(k)} x pseudo-group {int(g)} ({ne} errors / {ne + nc})"
            )
    if not dirs:
        raise ValueError(f"no (class x pseudo-group) cell with >= {min_errors} errors")
    d = F.normalize(torch.stack(dirs).float(), dim=-1)
    pooled = F.normalize(d.mean(0, keepdim=True), dim=-1)
    return torch.cat([pooled, d]), ["pooled"] + tags


def load_direction_file(path: str, rows: List[int] | None = None) -> torch.Tensor:
    """Orthonormal basis (r, D) of the error directions saved by
    discover_biases.py (<out stem>_dirs.npz), optionally only some rows."""
    d = torch.as_tensor(np.load(path)["dirs"]).float()
    if rows:
        d = d[rows]
    q, _ = torch.linalg.qr(d.T)
    return q.T.contiguous()


def load_core_dirs(path: str) -> torch.Tensor:
    """Accepted core directions (r, D) from a discover_core.py file."""
    with open(path) as f:
        entries = json.load(f)["directions"]
    dirs = [e["vector"] for e in entries if e.get("accept")]
    if not dirs:
        raise ValueError(f"no accepted core directions in {path}")
    return torch.tensor(dirs, dtype=torch.float32)


@torch.no_grad()
def prompt_directions(
    oracle: CLIPOracle, pairs: List[Tuple[List[str], List[str]]]
) -> torch.Tensor:
    """Written core concepts as CLIP directions (r, D): per (pos, neg) pair, the
    unit difference of the mean prompt embeddings.  Used like discovered core
    directions; scored residual to the concept subspace, the raw text margin's
    habitat/background component is removed."""
    dirs = [
        oracle.encode_text(pos).float().mean(0)
        - oracle.encode_text(neg).float().mean(0)
        for pos, neg in pairs
    ]
    return F.normalize(torch.stack(dirs).cpu(), dim=-1)
