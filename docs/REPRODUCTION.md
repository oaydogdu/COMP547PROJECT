# Reproduction Guide

This document describes how to rerun the Fashion-MNIST experiments reported in our COMP547 project. All timing numbers in the report used a **Google Colab NVIDIA A100** with CUDA synchronization in the eval scripts.

---

## Prerequisites

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src   # or prefix each command with PYTHONPATH=src
```

GPU strongly recommended. Fashion-MNIST ARPG training (~20 epochs) and FID (2048 samples × multiple conditions) take several hours on A100.

---

## Pipeline Overview

```
1. Train PixelCNN++     →  checkpoint + BPD + sequential latency
2. Train ARPG           →  checkpoint (best.pt)
3. K-sweep (eval_arpg)  →  sweep.json + sample grids
4. build_fashion_report →  tradeoff plots + CSV
5. compute_fid_fashion  →  FID JSONs (local train reference)
```

---

## Step 1 — PixelCNN++ Baseline

```bash
make pcnnpp-train-fashion
make pcnnpp-eval-fashion
```

**Outputs**

| Path | Content |
|------|---------|
| `results/pixelcnnpp_fashion_e20/checkpoints/best.pt` | Best checkpoint |
| `results/pixelcnnpp_fashion_e20/eval/fashion_eval.json` | Latency, throughput, BPD |
| `results/pixelcnnpp_fashion_e20/eval/fashion_grid.png` | Sample grid |

**Reported baseline:** test BPD **2.67**, latency **1642 ms/img**, FID **15.7**.

---

## Step 2 — ARPG Training

```bash
make arpg-train-fashion
```

Or with Google Drive backup (Colab):

```bash
PYTHONPATH=src python scripts/train_arpg.py \
  --dataset fashion_mnist \
  --save-dir results/arpg_fashion \
  --epochs 20 \
  --batch-size 16 \
  --drive-backup-dir /content/drive/MyDrive/comp547_outputs/arpg_fashion
```

**Outputs:** `results/arpg_fashion/checkpoints/best.pt`, `last.pt`, epoch checkpoints.

---

## Step 3 — K-Sweep (Speed)

```bash
make arpg-eval-fashion
```

Runs **11 values of K** × **3 schedules** (random, raster, row) = 33 conditions.

**Output:** `results/arpg_fashion/eval/sweep.json`

Each entry includes `K`, `schedule`, `latency_ms_per_image`, `throughput_img_per_s`, and a path to a 25-sample grid PNG.

---

## Step 4 — Presentation Plots

```bash
make fashion-report
```

**Requires:** baseline `fashion_eval.json` and ARPG `sweep.json`.

**Outputs under** `results/fashion_presentation/`:

- `tradeoff_speed.png`
- `quality_strip_random.png`, `quality_strip_row.png`, …
- `fashion_mnist_summary.csv`

---

## Step 5 — FID Evaluation

Fashion-MNIST precomputed clean-fid stats are unavailable (URL 404). The script exports the **training set to PNG** once and builds custom reference statistics.

### Baseline FID

```bash
PYTHONPATH=src python scripts/compute_fid_fashion.py \
  --model baseline \
  --checkpoint results/pixelcnnpp_fashion_e20/checkpoints/best.pt \
  --out-dir results/fid/baseline \
  --n-samples 2048 \
  --compute-fid \
  --out-json results/fid/baseline_fid.json
```

### ARPG FID (example: K=28 random)

```bash
PYTHONPATH=src python scripts/compute_fid_fashion.py \
  --model arpg \
  --checkpoint results/arpg_fashion/checkpoints/best.pt \
  --out-dir results/fid/arpg_random_K28 \
  --k 28 --schedule random \
  --n-samples 2048 \
  --compute-fid \
  --out-json results/fid/arpg_random_K28_fid.json
```

Repeat for `K=1` and `K=784` for report anchor points.

**Reported FID (random):** K=1 → 283.5, K=28 → 28.8, K=784 → 37.7, baseline → 15.7.

---

## Colab Notebook

For a single guided workflow (Drive restore, sweep, FID, backup):

```
notebooks/FashionMNIST_ARPG_Colab.ipynb
```

---

## CIFAR-10 (Co-author)

CIFAR experiments use the same `train_arpg.py` / `eval_arpg.py` with `--dataset cifar10` and 16×16 grayscale preprocessing inside `arpg_runner.py`. See co-author branch/notebook `notebooks/ARPG_notebook.ipynb` for the CIFAR sweep workflow.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: ARPG` | Set `PYTHONPATH=src` |
| CUDA dataloader hang on Colab | Use `--num-workers 0` (default in Makefile) |
| clean-fid Fashion-MNIST 404 | Script builds local ref in `data/fashion_mnist_train_ref/` |
| OOM during FID | Reduce batch generation size in script; default uses batches of 25 |

---

## Expected Runtime (A100, rough)

| Stage | Time |
|-------|------|
| PixelCNN++ train (20 ep) | ~2–3 h |
| ARPG train (20 ep) | ~2–3 h |
| K-sweep (33 cond.) | ~1–5 min |
| FID baseline (2048 img) | ~1 h |
| FID ARPG ×3 K values | ~1–2 h |
