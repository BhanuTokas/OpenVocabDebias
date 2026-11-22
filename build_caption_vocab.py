#!/usr/bin/env python
"""
build_caption_vocab.py — A concept vocabulary from image captions.

The phrases come from the data instead of a written list: a captioner (BLIP-2
by default) describes a run's train errors and an equal number of correctly
classified train images per class, and the noun phrases of the captions form a
vocabulary for discover_biases.py --vocab_file.  The captions are unprompted,
so nothing steers them toward backgrounds or people.

Also written, for comparison with Bias-to-Text (B2T, Kim et al.): each phrase's
CLIP similarity to a class's errors minus its similarity to that class's correct
images, the keyword score B2T ranks by.  Only target labels are used; captions
name the target class freely, so no phrase is removed here.

Usage
-----
    python build_caption_vocab.py <ERM run dir> --ckpt epoch_01.pt --out_dir results_captions/wb_resnet18
Outputs in --out_dir: vocab.txt (one phrase per line, most frequent first),
captions.jsonl, b2t.json.
"""

import argparse
import json
import os
import re
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

from clip_debias.clip_oracle import CLIPOracle
from clip_debias.discovery import as_prompt
from discover_biases import model_errors
from error_set import train_eval_dataset
from eval_checkpoints import load_run_cfg

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", help="run directory with config.json and --ckpt")
    p.add_argument("--ckpt", default="epoch_01.pt")
    p.add_argument("--celeba_root", default=None)
    p.add_argument("--captioner", default="Salesforce/blip2-opt-2.7b")
    p.add_argument(
        "--per_class",
        type=int,
        default=1000,
        help="at most this many errors per class, plus as many correct images",
    )
    p.add_argument("--min_count", type=int, default=2, help="drop rarer phrases")
    p.add_argument("--max_new_tokens", type=int, default=30)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def pick(y, err, per_class, rng):
    """Per class: up to per_class errors and the same number of correct images."""
    out = []
    for k in np.unique(y):
        e = np.flatnonzero((y == k) & err)
        c = np.flatnonzero((y == k) & ~err)
        e = rng.choice(e, min(len(e), per_class), replace=False)
        c = rng.choice(c, min(len(c), len(e)), replace=False)
        out += [(int(i), int(k), True) for i in e] + [
            (int(i), int(k), False) for i in c
        ]
    return out


def noun_phrases(caption, tagger):
    """Adjective/noun runs ending in a noun, and each noun on its own."""
    words = re.findall(r"[a-z]+(?:-[a-z]+)?", caption.lower())
    tags = tagger(words)
    phrases, run = set(), []
    for w, t in tags + [("", ".")]:
        if t.startswith(("NN", "JJ")):
            run.append((w, t))
            continue
        while run and not run[-1][1].startswith("NN"):
            run.pop()
        if run:
            phrases.add(" ".join(w for w, _ in run))
            phrases.update(w for w, t in run if t.startswith("NN"))
        run = []
    return phrases


@torch.no_grad()
def caption_images(images, args, device):
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    processor = Blip2Processor.from_pretrained(args.captioner)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.captioner, torch_dtype=torch.float16
    ).to(device)
    out = []
    for s in range(0, len(images), args.batch_size):
        batch = processor(images=images[s : s + args.batch_size], return_tensors="pt")
        batch = batch.to(device, torch.float16)
        ids = model.generate(**batch, max_new_tokens=args.max_new_tokens)
        out += [
            c.strip() for c in processor.batch_decode(ids, skip_special_tokens=True)
        ]
        print(f"  captioned {len(out):,} / {len(images):,}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    args = parse_args()
    cfg, dataset, _ = load_run_cfg(
        args.run_dir, SimpleNamespace(**vars(args), dataset="auto")
    )
    ds = train_eval_dataset(cfg, dataset)
    y, err = model_errors(args.run_dir, cfg, ds, np.arange(len(ds)), args)
    chosen = pick(y, err, args.per_class, np.random.default_rng(args.seed))
    for k in np.unique(y):
        n_e = sum(1 for _, c, e in chosen if c == k and e)
        print(f"  class {k}: captioning {n_e} errors and {n_e} correct images")

    tensors = torch.stack([ds[i][0] for i, _, _ in chosen])
    pil = [to_pil_image((x * STD + MEAN).clamp(0, 1)) for x in tensors]
    captions = caption_images(pil, args, cfg.device)

    import nltk

    for res in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
        nltk.download(res, quiet=True)
    counts, per_caption = Counter(), []
    for cap in captions:
        ph = noun_phrases(cap, nltk.pos_tag)
        per_caption.append(sorted(ph))
        counts.update(ph)
    vocab = [p for p, n in counts.most_common() if n >= args.min_count]
    print(f"  {len(counts)} phrases, {len(vocab)} seen at least {args.min_count} times")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(vocab) + "\n")
    with open(os.path.join(args.out_dir, "captions.jsonl"), "w", encoding="utf-8") as f:
        for (i, k, e), cap, ph in zip(chosen, captions, per_caption):
            f.write(
                json.dumps(dict(idx=i, y=k, error=e, caption=cap, phrases=ph)) + "\n"
            )

    # B2T-style keyword score per class, for comparison
    oracle = CLIPOracle(cfg.clip_model, device=cfg.device)
    v = torch.cat(
        [
            oracle.encode_images(oracle.preprocess(tensors[s : s + 256].to(cfg.device)))
            .float()
            .cpu()
            for s in range(0, len(tensors), 256)
        ]
    )
    t = oracle.encode_text([as_prompt(p) for p in vocab]).float().cpu()
    sim = (v @ t.T).numpy()
    ys = np.array([k for _, k, _ in chosen])
    es = np.array([e for _, _, e in chosen])
    b2t = {}
    for k in np.unique(ys):
        d = sim[(ys == k) & es].mean(0) - sim[(ys == k) & ~es].mean(0)
        order = np.argsort(-d)
        b2t[int(k)] = [[vocab[i], round(float(d[i]), 4)] for i in order[:30]]
        print(f"  class {k}, B2T top: " + ", ".join(vocab[i] for i in order[:10]))
    with open(os.path.join(args.out_dir, "b2t.json"), "w", encoding="utf-8") as f:
        json.dump(
            dict(captioner=args.captioner, ckpt=args.ckpt, scores=b2t), f, indent=1
        )
    print(f"  written to {args.out_dir}")


if __name__ == "__main__":
    main()
