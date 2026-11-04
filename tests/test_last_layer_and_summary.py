"""DFR helpers on synthetic features, and the run-name patterns of the summary."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from last_layer import balanced_subsample, dfr_train, fit_dfr, wga
from scripts.summarize_main import BAM, JTT


def test_balanced_subsample_equalises_cells():
    g = np.repeat([0, 1, 2, 3], [500, 20, 7, 300])
    idx = balanced_subsample(g, np.random.default_rng(0))
    assert np.bincount(g[idx]).tolist() == [7, 7, 7, 7]
    assert len(np.unique(idx)) == len(idx)


def test_wga_is_the_worst_group_accuracy():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 0])
    g = np.array([0, 0, 1, 1])
    worst, accs = wga(pred, y, g)
    assert worst == 0.5 and accs == {0: 1.0, 1: 0.5}


def spurious(n, p_align, rng):
    """Feature 0: noisy core signal; feature 1: clean spurious signal that agrees
    with y on a fraction p_align of the examples."""
    y = rng.integers(0, 2, n)
    a = np.where(rng.random(n) < p_align, y, 1 - y)
    x = np.stack(
        [y + 0.6 * rng.standard_normal(n), a + 0.05 * rng.standard_normal(n)], 1
    )
    return x.astype(np.float32), y, y * 2 + a


def test_dfr_on_balanced_train_cells_ignores_the_spurious_feature():
    rng = np.random.default_rng(0)
    xf, yf, gf = spurious(4000, 0.95, rng)  # biased train split
    xv, yv, gv = spurious(1000, 0.5, rng)
    xt, yt, gt = spurious(2000, 0.5, rng)
    pred, params = dfr_train(xf, yf, gf, xv, yv, gv, xt, rng)
    assert wga(pred, yt, gt)[0] > 0.7  # the spurious-only rule gets 0 on two groups
    assert set(params) == {"C", "val_select_wga"}


def test_dfr_fit_is_l1_sparse():
    # scikit-learn deprecates penalty="l1" (removed in 1.10): fail loudly if it
    # silently turns into L2 after an upgrade
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 50))
    y = (x[:, 0] > 0).astype(int)
    w, _ = fit_dfr(x, y, y, C=0.05, n_fits=1, rng=rng)
    assert (w == 0).sum() >= 40


@pytest.mark.parametrize(
    "name,ok",
    [
        ("bam2_aux50_T3_lr1e-5_up140", True),
        ("bam2_aux50_T3_up140", False),  # stage 1 at lr 1e-4 (overfit)
        ("bam2_aux5_up50", False),  # epoch-1 error set (= ERM)
    ],
)
def test_summary_only_reads_current_bam_runs(name, ok):
    assert bool(re.match(BAM, name)) == ok


def test_summary_jtt_pattern():
    assert re.match(JTT, "jtt_lr1e-5_up20").groups() == ("1e-5", "20")
    assert not re.match(JTT, "jtt_up20")


def test_select_lines_reach_a_child_shell(tmp_path):
    # submit_main.sh runs as a child of the shell that evals these lines, so
    # plain assignments would not reach it
    d = tmp_path / "results_x" / "wb_resnet18" / "jtt_lr1e-5_up20" / "seed_1"
    d.mkdir(parents=True)
    (d / "val_history.json").write_text('{"epochs": {"1": {"val_pseudo_wga": 0.5}}}')
    script = Path(__file__).parent.parent / "scripts" / "summarize_main.py"
    cmd = [sys.executable, str(script), "--root", "x", "--select"]
    out = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "export JTT_LR_wb_resnet18=1e-5 JTT_UP_wb_resnet18=20"
