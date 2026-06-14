"""Build presentation-ready plots and summary tables for Fashion-MNIST runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(
    baseline_eval_json: str,
    arpg_sweep_json: str,
    out_dir: str,
    baseline_metrics_json: str | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ar_eval = _load_json(Path(baseline_eval_json))
    sweep_data = _load_json(Path(arpg_sweep_json))
    sweep = sweep_data["sweep"]

    schedules = ["random", "raster", "row"]
    colors = {"random": "#2196F3", "raster": "#FF9800", "row": "#4CAF50"}
    markers = {"random": "o", "raster": "s", "row": "^"}
    labels = {"random": "Random (ARPG)", "raster": "Raster (structured)", "row": "Row-by-row"}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Fashion-MNIST: Speed–Quality Tradeoff vs AR Baseline", fontsize=14, fontweight="bold")

    for sched in schedules:
        rows = [r for r in sweep if r["schedule"] == sched]
        Ks = np.array([r["K"] for r in rows])
        lats = np.array([r["latency_ms_per_image"] for r in rows])
        tps = np.array([r["throughput_img_per_s"] for r in rows])
        axes[0].plot(Ks, lats, marker=markers[sched], color=colors[sched], label=labels[sched], linewidth=2)
        axes[1].plot(Ks, tps, marker=markers[sched], color=colors[sched], label=labels[sched], linewidth=2)

    ar_lat = ar_eval["latency_ms_per_image"]
    ar_tp = ar_eval["throughput_img_per_s"]
    axes[0].axhline(ar_lat, color="red", linestyle="--", linewidth=2, label=f"AR Sequential {ar_lat:.0f} ms")
    axes[1].axhline(ar_tp, color="red", linestyle="--", linewidth=2, label=f"AR Sequential {ar_tp:.1f} img/s")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlabel("K (number of decode steps; lower = faster)")
    axes[0].set_ylabel("Latency (ms / image)")
    axes[1].set_ylabel("Throughput (img / s)")
    plt.tight_layout()
    tradeoff_path = out / "tradeoff_speed.png"
    plt.savefig(tradeoff_path, dpi=180)
    plt.close()

    for sched in schedules:
        rows = [r for r in sweep if r["schedule"] == sched]
        fig2, axes2 = plt.subplots(1, len(rows), figsize=(2.8 * len(rows), 3.2))
        if len(rows) == 1:
            axes2 = [axes2]
        for ax, r in zip(axes2, rows):
            ax.imshow(Image.open(r["grid"]), cmap="gray")
            ax.axis("off")
            ax.set_title(f"K={r['K']} ({r['latency_ms_per_image']:.0f}ms)", fontsize=8)
        plt.suptitle(f"Schedule: {sched}", fontsize=12)
        plt.tight_layout()
        strip_path = out / f"quality_strip_{sched}.png"
        plt.savefig(strip_path, dpi=150)
        plt.close()

    rows_table = []
    for r in sweep:
        rows_table.append({
            "model": "ARPG",
            "schedule": r["schedule"],
            "K": r["K"],
            "latency_ms": round(r["latency_ms_per_image"], 2),
            "throughput_img_s": round(r["throughput_img_per_s"], 3),
        })
    rows_table.append({
        "model": "PixelCNN++",
        "schedule": "sequential",
        "K": 784,
        "latency_ms": round(ar_lat, 2),
        "throughput_img_s": round(ar_tp, 3),
    })
    df = pd.DataFrame(rows_table)
    csv_path = out / "fashion_mnist_summary.csv"
    df.to_csv(csv_path, index=False)

    rand_rows = [r for r in sweep if r["schedule"] == "random"]
    sweet = min(rand_rows, key=lambda r: abs(r["latency_ms_per_image"] - ar_lat * 0.25))
    speedup = ar_lat / sweet["latency_ms_per_image"]

    presentation = {
        "dataset": "fashion_mnist",
        "baseline": {
            "model": "PixelCNN++",
            "latency_ms_per_image": ar_lat,
            "throughput_img_per_s": ar_tp,
            "checkpoint": ar_eval.get("checkpoint"),
            "best_test_bpd": ar_eval.get("best_test_bpd"),
        },
        "arpg_sweet_spot_random": {
            "K": sweet["K"],
            "latency_ms_per_image": sweet["latency_ms_per_image"],
            "throughput_img_per_s": sweet["throughput_img_per_s"],
            "speedup_vs_baseline": round(speedup, 2),
        },
        "artifacts": {
            "tradeoff_speed_png": str(tradeoff_path),
            "summary_csv": str(csv_path),
        },
    }

    if baseline_metrics_json and Path(baseline_metrics_json).exists():
        hist = json.loads(Path(baseline_metrics_json).read_text())
        best = min(hist, key=lambda x: x["test_bpd"])
        presentation["baseline"]["best_test_bpd_from_history"] = best["test_bpd"]
        presentation["baseline"]["best_test_bpd_epoch"] = best["epoch"]

        epochs = [h["epoch"] for h in hist]
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, [h["train_bpd"] for h in hist], label="train")
        plt.plot(epochs, [h["test_bpd"] for h in hist], label="test")
        plt.xlabel("epoch")
        plt.ylabel("BPD")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.title("PixelCNN++ Training Curve (Fashion-MNIST)")
        bpd_path = out / "baseline_bpd_curve.png"
        plt.savefig(bpd_path, dpi=150)
        plt.close()
        presentation["artifacts"]["baseline_bpd_curve_png"] = str(bpd_path)

    pres_path = out / "presentation_summary.json"
    pres_path.write_text(json.dumps(presentation, indent=2), encoding="utf-8")
    print(json.dumps(presentation, indent=2))
    return presentation


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-eval-json", required=True)
    p.add_argument("--arpg-sweep-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--baseline-metrics-json", default=None)
    args = p.parse_args()
    build_report(
        baseline_eval_json=args.baseline_eval_json,
        arpg_sweep_json=args.arpg_sweep_json,
        out_dir=args.out_dir,
        baseline_metrics_json=args.baseline_metrics_json,
    )


if __name__ == "__main__":
    main()
