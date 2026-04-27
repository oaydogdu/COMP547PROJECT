"""CLI: K-sweep + decoding schedule comparison for PixelARPG."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ARPG.arpg_runner import run_arpg_sweep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir",    required=True)
    p.add_argument("--ks",
                   default="1,2,4,7,14,28,56,112,196,392,784",
                   help="comma-separated K values (number of decode steps)")
    p.add_argument("--schedules",
                   default="random,raster,row",
                   help="comma-separated: random | raster | row | column")
    p.add_argument("--n-samples",  type=int, default=25)
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    summary = run_arpg_sweep(
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        k_values=tuple(int(k) for k in args.ks.split(",")),
        schedules=tuple(args.schedules.split(",")),
        n_samples=args.n_samples,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
