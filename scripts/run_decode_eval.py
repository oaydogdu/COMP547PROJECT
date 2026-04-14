from __future__ import annotations

import argparse

from arpg.evaluate import evaluate_decode_modes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out-json", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--schedule", type=str, default="random", choices=["random", "block-raster"])
    p.add_argument("--block-size", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_decode_modes(
        checkpoint_path=args.checkpoint,
        out_json=args.out_json,
        batch_size=args.batch_size,
        schedule=args.schedule,
        block_size=args.block_size,
    )


if __name__ == "__main__":
    main()
