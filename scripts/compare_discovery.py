#!/usr/bin/env python
"""
compare_discovery.py — Agreement between discover_biases.py outputs.

For each class: the |cosine| between the discovered CLIP directions of every
pair of files (sign is arbitrary), the number of errors behind each direction,
the top phrases, and (if --report_truth was used) the correlation with the true
concept.

    python scripts/compare_discovery.py results_cf/wb_resnet18/*.json
"""

import argparse
import json
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="discover_biases.py --out files")
    a = p.parse_args()
    runs = {}
    for f in a.files:
        d = np.load(os.path.splitext(f)[0] + "_dirs.npz")
        props = {e["class"]: e for e in json.load(open(f))["proposals"]}
        runs[os.path.basename(f)[:-5]] = {
            int(k): (d["dirs"][i], int(d["n_errors"][i]), props[int(k)])
            for i, k in enumerate(d["classes"])
        }
    names = list(runs)
    w = max(len(n) for n in names)
    for k in sorted({k for r in runs.values() for k in r}):
        print(f"\n== class {k}")
        for n in names:
            if k not in runs[n]:
                print(f"  {n:<{w}}  (too few errors)")
                continue
            _, n_err, e = runs[n][k]
            truth = "  ".join(
                f"{t} {abs(c):.2f}" for t, c in e.get("truth_corr", {}).items()
            )
            strip = lambda s: s.replace("a photo of ", "")
            print(
                f"  {n:<{w}}  {n_err:6d} err  |corr| true {truth or '-':<14}  "
                f"{', '.join(map(strip, e['pos'][:3]))}  <->  "
                f"{', '.join(map(strip, e['neg'][:3]))}"
            )
        have = [n for n in names if k in runs[n]]
        if len(have) > 1:
            print("  |cosine| between directions:")
            print("  " + " " * w + "".join(f"{i:>7d}" for i in range(len(have))))
            for i, n in enumerate(have):
                cos = [abs(float(runs[n][k][0] @ runs[m][k][0])) for m in have]
                print(f"  {n:<{w}}" + "".join(f"{c:7.2f}" for c in cos) + f"  [{i}]")


if __name__ == "__main__":
    main()
