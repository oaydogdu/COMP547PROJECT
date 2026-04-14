from __future__ import annotations

import argparse

from arpg.train import TrainArgs, train_baseline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="fashion_mnist", choices=["fashion_mnist", "cifar10"])
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--out-checkpoint", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_baseline(
        TrainArgs(
            dataset=args.dataset,
            vocab_size=args.vocab_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            out_checkpoint=args.out_checkpoint,
        )
    )


if __name__ == "__main__":
    main()
