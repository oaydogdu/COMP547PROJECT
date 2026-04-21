# COMP547 Project: ARPG Speed-Quality Tradeoff

This repository contains the baseline stage for our COMP547 project:
a modernized PixelCNN++ autoregressive image generation pipeline.

## What We Are Building

- A reproducible PixelCNN++ baseline training/sampling pipeline.
- Fashion-MNIST and CIFAR-10 support.
- Evaluation outputs for efficiency (`latency`, `throughput`) and sample grids.

## Quick Structure

- `docs/` project scope, experiment protocol, reporting templates.
- `configs/` experiment configuration artifacts.
- `src/KlassikAR/` PixelCNN++ model, layers, loss/sampling utilities, train/eval runner.
- `scripts/` train/eval entrypoints for PixelCNN++.
- `results/` metrics, plots, and generated sample grids.

## Baseline Roadmap

1. Train PixelCNN++ baseline on Fashion-MNIST and CIFAR-10.
2. Validate checkpoints with sampling outputs and speed metrics.
3. Use validated baseline as the foundation for next-stage ARPG-style decoding work.

## Cursor + Lightning AI Workflow

Use Cursor as your IDE and Lightning AI as the GPU execution environment.

1. In Cursor (local), edit code and push to GitHub:
   - `git add .`
   - `git commit -m "your message"`
   - `git push origin <branch>`
2. In Lightning Studio terminal, pull latest code:
   - `git clone https://github.com/oaydogdu/COMP547PROJECT.git` (first time)
   - `cd COMP547PROJECT`
   - `git pull origin <branch>`
3. Install dependencies in Lightning:
   - `python -m pip install -r requirements.txt`
4. Run training/eval in Lightning:
   - `make pcnnpp-train-fashion`
   - `make pcnnpp-eval-fashion`

Notes:
- The Makefile sets `PYTHONPATH=src` so `arpg` imports work consistently.
- Keep all run outputs under `results/` and commit only code/config/docs (not large checkpoints).

## PixelCNN++ Baseline (Public Source Integration)

We integrated a modernized PixelCNN++ baseline derived from public source code so we can run stronger classical autoregressive image generation before ARPG-style parallel decoding experiments.

Train:
- `make pcnnpp-train-fashion`
- `make pcnnpp-train-cifar`

Evaluate and save sample grid + speed metrics:
- `make pcnnpp-eval-fashion`
- `make pcnnpp-eval-cifar`

Direct scripts:
- `scripts/train_pixelcnnpp.py`
- `scripts/eval_pixelcnnpp.py`
