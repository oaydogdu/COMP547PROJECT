from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment_matrix.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, choices=["fashion_mnist", "cifar10"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--python-bin", type=str, default="python")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    block_sizes = cfg["parallel_decode"]["block_sizes"][args.dataset]
    schedules = cfg["parallel_decode"]["schedules"]

    out_root = Path("results") / "minimum_matrix" / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    for schedule in schedules:
        for block_size in block_sizes:
            out_json = out_root / f"{schedule}_b{block_size}.json"
            cmd = [
                args.python_bin,
                "scripts/run_decode_eval.py",
                "--checkpoint",
                args.checkpoint,
                "--out-json",
                str(out_json),
                "--batch-size",
                str(args.batch_size),
                "--schedule",
                schedule,
                "--block-size",
                str(block_size),
            ]
            subprocess.run(cmd, check=True)

    merged = []
    for path in sorted(out_root.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            merged.append(json.load(f))
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)


if __name__ == "__main__":
    main()
