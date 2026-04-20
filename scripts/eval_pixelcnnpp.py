from __future__ import annotations

import argparse

from arpg.pixelcnnpp_runner import evaluate_pixelcnnpp_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out-json", type=str, required=True)
    p.add_argument("--out-grid", type=str, required=True)
    p.add_argument("--sample-batch-size", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_pixelcnnpp_checkpoint(
        checkpoint_path=args.checkpoint,
        out_json=args.out_json,
        out_grid=args.out_grid,
        sample_batch_size=args.sample_batch_size,
        seed=args.seed,
    )
    print(f"saved_eval={args.out_json}")
    print(f"saved_grid={args.out_grid}")


if __name__ == "__main__":
    main()
