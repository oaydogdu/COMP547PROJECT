from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-json", type=str, required=True)
    p.add_argument("--out-png", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.summary_json).open("r", encoding="utf-8") as f:
        rows = json.load(f)

    labels = []
    latencies = []
    throughputs = []
    for row in rows:
        labels.append(f'{row["schedule"]}-b{row["block_size"]}')
        latencies.append(row["parallel"]["latency_ms"])
        throughputs.append(row["parallel"]["throughput_img_s"])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(labels, latencies)
    ax[0].set_title("Parallel Decode Latency (ms)")
    ax[0].tick_params(axis="x", rotation=45)

    ax[1].bar(labels, throughputs)
    ax[1].set_title("Parallel Decode Throughput (img/s)")
    ax[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=160)


if __name__ == "__main__":
    main()
