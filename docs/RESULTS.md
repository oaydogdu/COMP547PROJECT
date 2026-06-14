# Results Summary

Fashion-MNIST numbers from our final A100 runs (June 2026). Checkpoints and JSON logs were backed up to Google Drive under `comp547_outputs/final_20260609_1529/`.

---

## Sequential Baseline (PixelCNN++)

| Metric | Value |
|--------|-------|
| Test BPD (best) | 2.67 |
| Latency | 1642 ms / image |
| Throughput | 0.61 img / s |
| FID (n=2048) | 15.7 |

Checkpoint: `pixelcnnpp_fashion_e20/checkpoints/best.pt`

---

## ARPG Random Schedule

### Headline operating points

| K | Latency | Throughput | FID | ΔFID vs baseline |
|---|---------|------------|-----|------------------|
| 1 | 0.9 ms | 1140.8 img/s | 283.5 | +267.8 |
| 28 | 19.3 ms | 51.9 img/s | 28.8 | +13.1 |
| 784 | 425.4 ms | 2.4 img/s | 37.7 | +22.0 |

Test BPD for ARPG: **~2.72** (stable across K).

### Full latency sweep (random)

| K | ms/img | img/s |
|---|--------|-------|
| 1 | 0.9 | 1140.8 |
| 2 | 1.7 | 591.1 |
| 4 | 3.4 | 297.4 |
| 7 | 5.5 | 180.7 |
| 14 | 9.7 | 103.6 |
| 28 | 19.3 | 51.9 |
| 56 | 37.8 | 26.4 |
| 112 | 75.0 | 13.3 |
| 196 | 127.9 | 7.8 |
| 392 | 241.9 | 4.1 |
| 784 | 425.4 | 2.4 |

Source: `results/arpg_fashion/eval/sweep.json` (schedule = random).

---

## Artifacts

| Artifact | Typical path |
|----------|--------------|
| Baseline eval JSON | `results/pixelcnnpp_fashion_e20/eval/fashion_eval.json` |
| ARPG sweep JSON | `results/arpg_fashion/eval/sweep.json` |
| Tradeoff plot | `results/fashion_presentation/tradeoff_speed.png` |
| Quality strips | `results/fashion_presentation/quality_strip_*.png` |
| FID JSONs | `results/fid/baseline_fid.json`, `results/fid/arpg_random_K*_fid.json` |

---

## Takeaways

1. **Speed:** ARPG at K=784 matches the baseline’s 784 decode steps but runs **~3.9× faster** per image (425 ms vs 1642 ms).
2. **Quality:** FID never beats PixelCNN++; usable samples appear around **K≥28**, but FID stays **13–22 points** above baseline.
3. **Likelihood:** BPD alone is misleading—parallel decoding can match test BPD while FID degrades sharply (especially at K=1).

CIFAR-10 results are documented in the course report (Section 4.3, co-author).
