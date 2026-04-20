# COMP547 Project: ARPG Speed-Quality Tradeoff

This repository implements a small-scale study of randomized parallel decoding
for autoregressive image generation.

## What We Are Building

- A reproducible baseline autoregressive training/sampling pipeline.
- A randomized parallel decoding sampler with configurable schedules.
- A cost-aware experiment runner for Fashion-MNIST and CIFAR-10.
- Evaluation outputs for quality (`FID`) and efficiency (`latency`, `throughput`).

## Quick Structure

- `docs/` project scope, experiment protocol, reporting templates.
- `configs/` dataset/model/experiment configurations.
- `src/arpg/` core code (data, model, train, decode, evaluate).
- `scripts/` simple commands for setup, training, experiments, and plotting.
- `results/` metrics, plots, and generated sample grids.

## Execution Roadmap

1. Freeze scope and minimum experiment matrix (`docs/scope_freeze.md`).
2. Train baseline and validate sequential decoding.
3. Integrate randomized parallel decoding and correctness checks.
4. Run pilot + main ablations under cost constraints.
5. Produce speed-quality tradeoff figures and final report assets.

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
   - `make train-fashion`
   - `make eval-fashion`
   - `make matrix-fashion`

Notes:
- The Makefile sets `PYTHONPATH=src` so `arpg` imports work consistently.
- Keep all run outputs under `results/` and commit only code/config/docs (not large checkpoints).
