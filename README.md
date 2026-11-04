# OpenVocabDebias
Experiments for Debiasing Open Vocab concepts in Embeddings (DOVE). Zero-shot concept scrubbing for image classifiers using CLIP as a concept oracle. No labeled concept dataset required.

## Method summary

DOVE removes a spurious concept from an image classifier using CLIP as a concept
oracle.  No group or bias labels are used at any stage: the bias can be found
from the classifier's own errors, and models are selected without bias labels.

1. **Discover the bias** (`discover_biases.py`).  Train ERM for one epoch and
   take its errors on the train split.  Within each class, the direction
   `mean(CLIP of errors) − mean(CLIP of correct)` points along what the model
   gets wrong.  Each direction is scored on its own and named by the phrases of a
   vocabulary whose zero-shot scores correlate with it most.  The result is a
   pair of prompt sets, e.g. "a forest, a garden" vs "the ocean, the sea".
2. **Review the prompts.**  A person accepts each proposal in the output JSON,
   and deletes wrong phrases, such as words for the target class.  Hand-written
   prompts can be used instead.
3. **Train with the prompts.**  The prompts give CLIP concept scores and
   zero-shot pseudo-groups `(class, concept)`, used in two ways:
   - **Whitened decorrelation** (`--repulse whitened`) penalises the squared
     correlation between backbone features `E(x)` and the whitened concept scores.
   - **Pseudo-group reweighting** (`--group_reweight pseudo`) balances the groups.
4. **Select without bias labels.**  `best_pseudo_wga.pt` is the epoch with the
   best worst-group accuracy on val under the pseudo-groups.  It is the checkpoint
   each run reports.  `best_wga.pt`, selected with true val groups, is saved only
   as a reference.

Val is used only for checkpoint selection.  Earlier variants (orthogonal
projection of the CLIP target, distillation losses) are still available through
`--runs`, see `RUN_FACTORIES` in `train.py`.

## Setup

```bash
# 1. Install dependencies (uv.lock is the canonical environment)
uv sync

# 2. Download CelebA
#    Set download=True in data.py CelebA constructor on first run
#    Waterbirds is loaded through Hugging Face datasets
```

## Usage

Waterbirds, from nothing to a debiased model:

```bash
# Baseline: its epoch_01.pt is the model discovery reads
python waterbirds_train.py --runs erm --no_lambda_schedules --seeds 42

# Discover (use the run directory printed by training)
python discover_biases.py <ERM run dir> --split train --ckpt epoch_01.pt \
    --prompted none --out proposals.json

# Review proposals.json: fix phrases and set "accept": true

# Train with the accepted prompts
python waterbirds_train.py --runs repulse_only --repulse whitened --lambda_repulse 10 \
    --group_reweight pseudo --no_lambda_schedules --prompts none \
    --concept_subspace_k 3 --extra_prompts proposals.json

# Evaluate every selection rule on test
python eval_checkpoints.py <run dir> --out eval_summary.csv
```

CelebA uses `train.py`, whose flags differ a little: `scripts/train_confirmed.sh`
has both commands.  `prompts/*_ep1_train_seed42.json` are the reviewed prompts
discovered from epoch-1 train errors with this recipe, from seed 42.

### Reproducing the main tables

Run on a SLURM cluster, from the repo root on the login node:

```bash
PHASE=grid bash scripts/submit_main.sh          # every method, seeds 42 123 456
eval "$(python scripts/summarize_main.py --select)"
PHASE=extend bash scripts/submit_main.sh        # seeds 789 1011 at the chosen values
python scripts/summarize_main.py                # tables
```

`scripts/submit_main.sh` describes the phases.  The comparison covers ERM, JTT,
BAM, GroupDRO on pseudo and true groups, DFR and AFR (`last_layer.py`) and frozen
CLIP probes.  True groups are used only in rows marked as reference.

### Discovery experiments

- `scripts/discovery_schemes.sh` compares where the errors come from.  Each model
  trains on 0.8n train examples and is scored only on examples it did not draw:
  single holdout, single bootstrap, repeated holdout, 5-fold and repeated
  bootstrap, each at epoch 1 and the last epoch.
- `--vocab_file` replaces the built-in vocabulary.  `prompts/vocab/` holds lists
  written by Claude agents without project context (see its README), and
  `scripts/discovery_vocab.sh` compares them on the same errors.

Limitations: the names depend on the vocabulary, a person reviews the prompts, and
`prompts/*_seed42.json` come from one seed and are reused for the other seeds.

## Tests

CPU-only unit tests for the losses, GroupDRO, the K-fold split, discovery and DFR,
on synthetic tensors (no datasets or CLIP weights):

```bash
uv run --locked pytest    # or: python -m pytest
```

`uv.lock` is the canonical environment (used on Sol); `requirements.txt` lists the
same core packages for pip installs.

## Notes

- The decorrelation loss uses squared correlations, so it targets both positive and negative concept correlations
- CLIP (`openai/clip-vit-base-patch32`) is fully frozen throughout training
- Images are normalised with ImageNet stats for the backbone and re-normalised on-the-fly for CLIP (see `clip_preprocess.py`)
- AMP (automatic mixed precision) is enabled by default for consumer GPUs; disable with `--no_amp`
- Three metrics are reported: concept probe accuracy (target ~50%), overall task accuracy, and worst-group accuracy across (target × concept) subgroups; test metrics use the true groups

