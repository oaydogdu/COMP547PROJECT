# Scope Freeze

## One-Sentence Objective

Determine when ARPG-style randomized parallel decoding provides practically
useful inference speedup without unacceptable quality degradation in a
single-GPU, student-budget setting.

## Fixed Success Metrics

- Quality: `FID` (lower is better) + qualitative sample grids.
- Efficiency: generation `latency` and `throughput` (images/second).
- Reproducibility: fixed seeds, logged configs, and checkpointed runs.

## Included Scope

- Small-scale reproduction of randomized parallel decoding.
- Datasets: Fashion-MNIST (debug) and CIFAR-10 (main benchmark).
- Controlled extension on three axes:
  1. Parallel decoding level
  2. Decoding schedule
  3. Dataset complexity

## Excluded Scope

- ImageNet/full-scale reproduction.
- Proposing a new model architecture.
- Large hyperparameter sweeps not tied to core research questions.

## Minimum Delivery Matrix (Guaranteed)

- Fashion-MNIST:
  - Baseline sequential decoding
  - 2 parallelism levels
  - 2 schedules (`random`, `block-raster`)
- CIFAR-10:
  - Baseline sequential decoding
  - 3 parallelism levels
  - 2 schedules (`random`, `block-raster`)

Each condition must include: `FID`, `latency`, `throughput`, sample grid.

## Compute Policy (Cost-Aware)

- Default runtime: Colab Pro with T4.
- Use stronger GPU only for critical final runs or bottleneck checks.
- Run short pilots before expensive sweeps.
