from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-json", type=str, required=True)
    p.add_argument("--out-md", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.summary_json).open("r", encoding="utf-8") as f:
        rows = json.load(f)

    best_latency = min(rows, key=lambda r: r["parallel"]["latency_ms"])
    best_throughput = max(rows, key=lambda r: r["parallel"]["throughput_img_s"])

    lines = [
        f"# Tradeoff Summary ({args.dataset})",
        "",
        "## Best latency configuration",
        f'- schedule: `{best_latency["schedule"]}`',
        f'- block_size: `{best_latency["block_size"]}`',
        f'- latency_ms: `{best_latency["parallel"]["latency_ms"]:.2f}`',
        "",
        "## Best throughput configuration",
        f'- schedule: `{best_throughput["schedule"]}`',
        f'- block_size: `{best_throughput["block_size"]}`',
        f'- throughput_img_s: `{best_throughput["parallel"]["throughput_img_s"]:.2f}`',
        "",
        "## Notes",
        "- FID values should be merged here once computed from sample sets.",
        "- Compare against sequential baseline before final claims.",
    ]

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
