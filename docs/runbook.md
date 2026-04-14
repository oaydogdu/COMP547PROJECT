# End-to-End Runbook

## Environment

1. Install deps:
   - `python3 -m pip install -r requirements.txt`
2. Set package path:
   - `export PYTHONPATH=src`

## Phase A: Baseline

- Fashion-MNIST baseline:
  - `python3 scripts/train_baseline.py --dataset fashion_mnist --epochs 1 --out-checkpoint results/checkpoints/fashion_baseline.pt`
- CIFAR-10 baseline:
  - `python3 scripts/train_baseline.py --dataset cifar10 --epochs 1 --out-checkpoint results/checkpoints/cifar10_baseline.pt`

## Phase B: Decode Correctness + Speed

- Smoke test:
  - `python3 scripts/smoke_test_decode.py`
- Single decode comparison:
  - `python3 scripts/run_decode_eval.py --checkpoint results/checkpoints/fashion_baseline.pt --out-json results/eval/fashion_random_b16.json --schedule random --block-size 16`

## Phase C: Minimum Matrix Ablation

- Fashion-MNIST matrix:
  - `python3 scripts/run_minimum_matrix.py --dataset fashion_mnist --checkpoint results/checkpoints/fashion_baseline.pt`
- CIFAR-10 matrix:
  - `python3 scripts/run_minimum_matrix.py --dataset cifar10 --checkpoint results/checkpoints/cifar10_baseline.pt`

## Phase D: Tradeoff Outputs

- Plot speed tradeoff:
  - `python3 scripts/plot_tradeoff.py --summary-json results/minimum_matrix/cifar10/summary.json --out-png results/plots/cifar10_tradeoff.png`
- Build markdown summary:
  - `python3 scripts/build_report.py --summary-json results/minimum_matrix/cifar10/summary.json --out-md results/reports/cifar10_summary.md --dataset cifar10`

## Phase E: Final Narrative

- Merge FID values into JSON summaries.
- Use `docs/final_findings_template.md` to write the final conclusions.
