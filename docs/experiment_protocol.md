# Experiment Protocol

## 1) Baseline Training

Run on Fashion-MNIST first:

`python scripts/train_baseline.py --dataset fashion_mnist --epochs 1 --out-checkpoint results/checkpoints/fashion_baseline.pt`

Then run CIFAR-10 baseline:

`python scripts/train_baseline.py --dataset cifar10 --epochs 1 --out-checkpoint results/checkpoints/cifar10_baseline.pt`

Increase epochs for real runs after sanity checks pass.

## 2) Correctness and Speed Check

Run decode comparison on each checkpoint:

`python scripts/run_decode_eval.py --checkpoint results/checkpoints/fashion_baseline.pt --out-json results/eval/fashion_random_b16.json --schedule random --block-size 16`

## 3) Minimum Matrix

Fashion-MNIST:

`python scripts/run_minimum_matrix.py --dataset fashion_mnist --checkpoint results/checkpoints/fashion_baseline.pt`

CIFAR-10:

`python scripts/run_minimum_matrix.py --dataset cifar10 --checkpoint results/checkpoints/cifar10_baseline.pt`

## 4) Plot Tradeoff

`python scripts/plot_tradeoff.py --summary-json results/minimum_matrix/cifar10/summary.json --out-png results/plots/cifar10_tradeoff.png`

## 5) Reporting

For each run save:
- config snapshot
- checkpoint path
- timing JSON
- sample grid image
- optional FID score file
