#!/usr/bin/env python
"""
summarize_main.py — Main tables from scripts/submit_main.sh outputs.

Every trained method is read at best_pseudo_wga.pt (label-free selection).
Grid families (jtt_up*, bam2_aux*_up*) are reduced to one configuration per
(dataset, backbone) by the mean over seeds of the best val pseudo-group WGA
(val_history.json): no test or bias labels are used for any choice.  DFR / AFR
rows come from last_layer.py, the frozen-CLIP rows from clip_probe.py.

    python scripts/summarize_main.py             # tables
    python scripts/summarize_main.py --select    # chosen JTT / BAM values, as
                                                 # env lines for PHASE=extend
    python scripts/summarize_main.py --grid      # every grid configuration too
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

LABEL = {
    "erm": "ERM",
    "grw": "CLIP pseudo-group reweighting",
    "ours": "ours (decorrelation + reweighting)",
    "dro_pseudo": "GroupDRO, CLIP pseudo-groups (B2T-style)",
    "dro_oracle": "GroupDRO, true groups (reference)",
    "dro_pseudo_wd": "GroupDRO, CLIP pseudo-groups, strong weight decay",
    "dro_oracle_wd": "GroupDRO, true groups, strong weight decay (reference)",
    "jtt": "JTT",
    "bam2": "BAM",
    "dfr_pseudo": "DFR, CLIP pseudo-groups",
    "dfr_true": "DFR, true groups (reference)",
    "afr_pseudo": "AFR, selected on CLIP pseudo-groups",
    "afr_true": "AFR, selected on true groups (reference)",
}


JTT = r"jtt_lr([\d.e-]+)_up([\d.]+)$"
BAM = r"bam2_aux([\d.]+)_up([\d.]+)$"
EXACT = {  # run directories shown as one row each (no grid selection)
    "erm": r"erm",
    "grw": r"grw",
    "ours": r"ours_l[\d.]+",
    "dro_pseudo": r"dro_pseudo",
    "dro_oracle": r"dro_oracle",
    "dro_pseudo_wd": r"dro_pseudo_wd[\d.]+",
    "dro_oracle_wd": r"dro_oracle_wd[\d.]+",
}


def mean_sd(v):
    return f"{st.mean(v):.3f} ± {st.pstdev(v):.3f}" if v else "-"


def val_score(method_dir):
    """Mean over seeds of the best val pseudo-WGA over epochs."""
    s = []
    for f in glob.glob(f"{method_dir}/seed_*/val_history.json"):
        v = [e["val_pseudo_wga"] for e in json.load(open(f))["epochs"].values()
             if e.get("val_pseudo_wga") is not None]
        if v:
            s.append(max(v))
    return (st.mean(s), len(s)) if s else (float("-inf"), 0)


def eval_rows(method_dir):
    out = []
    for f in glob.glob(f"{method_dir}/seed_*/eval_summary.csv"):
        out += [r for r in csv.DictReader(open(f)) if r["checkpoint"] == "best_pseudo_wga.pt"]
    return out


def line(name, rows, keys):
    cells = []
    for k in keys:
        v = [float(r[k]) for r in rows if r.get(k) not in (None, "")]
        cells.append(f"{k}: {mean_sd(v)}")
    print(f"    {name:<44} " + "   ".join(cells) + f"   n={len(rows)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="main")
    p.add_argument("--select", action="store_true")
    p.add_argument("--grid", action="store_true")
    a = p.parse_args()
    base = f"results_{a.root}"
    keys = ["worst_group_acc", "task_test_acc"]

    for tag_dir in sorted(glob.glob(f"{base}/*_resnet*")):
        tag = os.path.basename(tag_dir)
        methods = {os.path.basename(d): d for d in glob.glob(f"{tag_dir}/*")
                   if os.path.isdir(d) and os.path.basename(d) != "error_sets"
                   and not os.path.basename(d).startswith("last_layer")}
        chosen = {}
        for fam, pat in (("jtt", JTT), ("bam2", BAM)):
            cands = {n: val_score(d) for n, d in methods.items() if re.match(pat, n)}
            if cands:
                best = max(cands, key=lambda n: cands[n][0])
                chosen[fam] = (best, cands)
        if a.select:
            for fam, (best, _) in chosen.items():
                if fam == "jtt":
                    lr, up = re.match(JTT, best).groups()
                    print(f"JTT_LR_{tag}={lr} JTT_UP_{tag}={up}")
                else:
                    aux, up = re.match(BAM, best).groups()
                    print(f"BAM_AUX_{tag}={aux} BAM_UP_{tag}={up}")
            continue

        print(f"\n== {tag}  (test, best_pseudo_wga.pt; mean ± sd over seeds)")
        for key in ("erm", "grw", "dro_pseudo", "dro_pseudo_wd", "dro_oracle", "dro_oracle_wd",
                    "jtt", "bam2", "ours"):
            if key in chosen:
                name = chosen[key][0]
                line(f"{LABEL[key]} [{name}]", eval_rows(methods[name]), keys)
                if a.grid:
                    for n, (score, k) in sorted(chosen[key][1].items()):
                        print(f"        grid {n:<22} val pseudo-WGA {score:.4f} (n={k})"
                              f"   test WGA {mean_sd([float(r['worst_group_acc']) for r in eval_rows(methods[n])])}")
            else:
                for n in sorted(methods):
                    if re.fullmatch(EXACT[key], n):
                        line(f"{LABEL[key]} [{n}]", eval_rows(methods[n]), keys)
        for ll_dir in sorted(glob.glob(f"{tag_dir}/last_layer*")):
            src = os.path.basename(ll_dir)[len("last_layer_"):] or "erm"
            ll = defaultdict(list)
            for f in glob.glob(f"{ll_dir}/seed_*.csv"):
                for r in csv.DictReader(open(f)):
                    ll[r["run_name"]].append(r)
            for n in ("dfr_pseudo", "afr_pseudo", "dfr_true", "afr_true"):
                if ll.get(n):
                    line(f"{LABEL[n]} on {src}", ll[n], keys)

    if not a.select:
        for f in sorted(glob.glob(f"{base}/*_clip_probe.csv")):
            print(f"\n== {os.path.basename(f)[:-4]}  (frozen CLIP, seed-independent)")
            for r in csv.DictReader(open(f)):
                print(f"    {r['run_name']:<44} worst_group_acc: {float(r['worst_group_acc']):.3f}"
                      f"   task_test_acc: {float(r['task_test_acc']):.3f}   {r['params']}")


if __name__ == "__main__":
    main()
