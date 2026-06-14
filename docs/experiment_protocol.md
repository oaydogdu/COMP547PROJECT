# Experiment Protocol

See **[REPRODUCTION.md](REPRODUCTION.md)** for the up-to-date step-by-step workflow.

## Metrics (fixed for all runs)

| Metric | Definition |
|--------|------------|
| **Latency** | Wall-clock ms per generated image (CUDA sync in eval scripts) |
| **Throughput** | Images per second |
| **BPD** | Bits per dimension on test set |
| **FID** | clean-fid vs. local Fashion-MNIST train reference (n=2048) |

## Fashion-MNIST matrix

- **Baseline:** PixelCNN++, 20 epochs
- **ARPG:** Transformer, 20 epochs
- **K sweep:** {1, 2, 4, 7, 14, 28, 56, 112, 196, 392, 784}
- **Schedules:** random, raster, row

## Outputs per condition

Save under `results/`:

- Config / checkpoint path
- Timing JSON (`sweep.json` or eval JSON)
- Sample grid PNG
- FID JSON (selected K values)

## CIFAR-10

Co-author runs use `--dataset cifar10` with 16×16 grayscale and extended training (40 epochs). Details in course report Section 4.3.
