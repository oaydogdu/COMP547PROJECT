# Understanding the Speed–Quality Tradeoff of Randomized Parallel Decoding in Autoregressive Image Generation

**COMP547 Deep Unsupervised Learning · Koç University · Spring 2026**

**Authors:** [Oğuzhan Aydoğdu](https://github.com/oaydogdu) · Mehmet Kaan Kütük  
**Course project report:** randomized parallel decoding (ARPG-style) vs. sequential PixelCNN++ on small-scale image generation.

---

## Overview

Autoregressive image models generate sharp samples but decode **one pixel at a time**, which makes inference slow. We study whether **parallel decode steps**—revealing groups of pixels per forward pass under random or structured schedules—can reduce latency without unacceptable quality loss.

This repository provides:

- A **PixelCNN++** sequential baseline (Fashion-MNIST, 28×28 grayscale)
- A **Transformer-based ARPG** model with random-mask training and variable-step parallel decoding
- Scripts for **K-sweeps**, **FID evaluation**, tradeoff plots, and Colab notebooks

**Research question:** *When does randomized parallel decoding provide meaningful inference speedup without unacceptable quality loss in small-scale autoregressive image generation?*

---

## Key Results (Fashion-MNIST, Colab A100)

Random-schedule ARPG vs. our PixelCNN++ baseline:

| Setting | Latency | Throughput | FID ↓ | Notes |
|---------|---------|------------|-------|-------|
| **PixelCNN++** (sequential) | 1642 ms/img | 0.61 img/s | **15.7** | Quality reference |
| **ARPG** K=28 | 19.3 ms/img | 51.9 img/s | 28.8 | Balanced operating point |
| **ARPG** K=784 | 425 ms/img | 2.4 img/s | 37.7 | ~**3.9×** faster than baseline at same step count |
| **ARPG** K=1 | 0.9 ms/img | 1141 img/s | 283.5 | Extreme speed; samples are noise |

Likelihood (test BPD) stays near **2.7** for ARPG across K, so **FID** is needed to expose the quality cost. Full sweep tables and figures are in our course report; CIFAR-10 extension by co-author.

> **Note on K:** In our code, **K = number of decode steps** (forward passes), not “pixels per step”. K=1 is one pass over all pixels; K=784 is 784 refinement steps—the same count as sequential PixelCNN++, but each ARPG step uses a lighter Transformer forward pass.

---

## Repository Layout

```
COMP547PROJECT/
├── src/
│   ├── KlassikAR/          # PixelCNN++ model, training, sequential sampling
│   ├── ARPG/               # Transformer ARPG model, parallel decode, K-sweep
│   └── common/             # Checkpoint save/resume utilities
├── scripts/
│   ├── train_pixelcnnpp.py
│   ├── eval_pixelcnnpp.py
│   ├── train_arpg.py
│   ├── eval_arpg.py        # K × schedule sweep → sweep.json
│   ├── build_fashion_report.py
│   └── compute_fid_fashion.py
├── notebooks/
│   ├── FashionMNIST_ARPG_Colab.ipynb   # Fashion-MNIST: baseline + ARPG + FID (recommended)
│   └── ARPG_notebook.ipynb             # CIFAR-10 workflow (co-author)
├── configs/
│   └── experiment_matrix.yaml
├── docs/
│   ├── README.md           # Documentation index
│   ├── REPRODUCTION.md     # Step-by-step rerun instructions
│   ├── RESULTS.md          # Result summary & artifact paths
│   └── experiment_protocol.md
├── Makefile                # Common train/eval/report targets
└── requirements.txt
```

Checkpoints, large outputs, and datasets are **not** committed (`results/`, `data/` are gitignored). Restore from your own training runs or Google Drive backups.

---

## Quick Start

### 1. Environment

```bash
git clone https://github.com/oaydogdu/COMP547PROJECT.git
cd COMP547PROJECT
python -m pip install -r requirements.txt
```

Requires **Python 3.10+**, **PyTorch** with CUDA for training/eval at scale.

### 2. Colab (recommended)

Open [`notebooks/FashionMNIST_ARPG_Colab.ipynb`](notebooks/FashionMNIST_ARPG_Colab.ipynb): Drive mount → clone repo → restore checkpoints → K-sweep → FID → plots.

### 3. Local / server (Fashion-MNIST)

```bash
# Sequential baseline
make pcnnpp-train-fashion
make pcnnpp-eval-fashion

# ARPG + 33-condition sweep (11 K × 3 schedules)
make arpg-train-fashion
make arpg-eval-fashion

# Tradeoff plots + CSV
make fashion-report

# FID (2048 samples; builds local train reference on first run)
PYTHONPATH=src python scripts/compute_fid_fashion.py \
  --model baseline --checkpoint results/pixelcnnpp_fashion_e20/checkpoints/best.pt \
  --out-dir results/fid/baseline --compute-fid --out-json results/fid/baseline_fid.json
```

See [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) for full commands, expected outputs, and CIFAR-10 notes.

---

## Models & Methods

| Component | Description |
|-----------|-------------|
| **PixelCNN++** | Causal autoregressive baseline; 5 ResNet blocks, 160 filters, logistic mixture head |
| **ARPG** | 6-layer Transformer encoder (d=192, 6 heads); bidirectional attention; random mask ratio during training |
| **Decoding** | Arccos schedule splits pixels into K blocks; nucleus sampling (p=0.9); schedules: **random**, **raster**, **row** |
| **Metrics** | Latency, throughput, test BPD, FID (clean-fid with **local** Fashion-MNIST train reference) |

PixelCNN++ integration follows the public [pixel-cnn-pp](https://github.com/openai/pixel-cnn-pp) reference (see `external/`, not vendored in git).

---

## Citation

If you use this code or results, please cite our COMP547 project report (2026):

```bibtex
@misc{aydogdu2026arpg,
  title  = {Understanding the Speed--Quality Tradeoff of Randomized Parallel Decoding in Autoregressive Image Generation},
  author = {Aydo{\c{g}}du, O{\u{g}}uzhan and K{\"u}t{\"u}k, Mehmet Kaan},
  year   = {2026},
  note   = {COMP547 Deep Unsupervised Learning, Ko{\c{c}} University},
  url    = {https://github.com/oaydogdu/COMP547PROJECT}
}
```

---

## License

MIT License — see [LICENSE](LICENSE). Fashion-MNIST and CIFAR-10 datasets remain subject to their original terms.

---

## Acknowledgements

Course project for **COMP547** at Koç University. PixelCNN++ baseline derived from OpenAI’s reference implementation. FID via [clean-fid](https://github.com/GaParmar/clean-fid).
