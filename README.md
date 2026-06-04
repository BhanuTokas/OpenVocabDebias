# OpenVocabDebias
Experiments for Debiasing Open Vocab concepts in Embeddings (DOVE). Zero-shot concept scrubbing for image classifiers using CLIP as a concept oracle. No labeled concept dataset required.

## Method summary

Given a pre-trained classifier with penultimate activations `E(x)`:

1. **Concept direction** — encode a prompt ensemble with frozen CLIP text encoder and compute a directional difference vector `V_T` (e.g. male − female direction).
2. **Orthogonal projection** — for each image, compute `V_I_perp = V_I − (V_I · V̂_T) V̂_T` using the frozen CLIP image encoder.  This is the concept-scrubbed distillation target.
3. **Combined loss** — fine-tune the classifier with three terms:
   - `L_task` (cross-entropy) — preserves downstream accuracy
   - `L_align` (cosine distance to `V_I_perp`) — pulls representations toward scrubbed target
   - `L_repulse` (squared cosine similarity with `V_T`) — explicitly pushes away from concept direction


## Setup

```bash
# 1. Install dependencies
pip install -e .

# 2. Download CelebA
#    Set download=True in data.py CelebA constructor on first run
```

## Training

```bash
# Default: ResNet-50, target=Attractive, concept=Male (gender), 10 epochs
python train.py

# Custom options
python train.py \
  --celeba_root ./data/celeba \
  --backbone resnet50 \
  --target_attr Attractive \
  --concept_attr Male \
  --epochs 15 \
  --batch_size 64 \
  --lambda_task 1.0 \
  --lambda_align 1.0 \
  --lambda_repulse 0.5
```

## Evaluation only

```bash
python eval.py --checkpoint checkpoints/best.pt
```

## Notes

- The repulse term penalises `(P(E(x)) · V̂_T)²`, so it targets both positive and negative concept correlations
- CLIP (`openai/clip-vit-base-patch32`) is fully frozen throughout training
- Images are normalised with ImageNet stats for the backbone and re-normalised on-the-fly for CLIP (see `clip_preprocess.py`)
- AMP (automatic mixed precision) is enabled by default for consumer GPUs; disable with `--no_amp`