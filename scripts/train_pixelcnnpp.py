from __future__ import annotations

import argparse

from arpg.pixelcnnpp_runner import PixelCNNPPTrainArgs, train_pixelcnnpp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="fashion_mnist", choices=["fashion_mnist", "mnist", "cifar10"])
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--save-dir", type=str, default="results/pixelcnnpp")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-decay", type=float, default=0.999995)
    p.add_argument("--nr-resnet", type=int, default=5)
    p.add_argument("--nr-filters", type=int, default=160)
    p.add_argument("--nr-logistic-mix", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sample-batch-size", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = train_pixelcnnpp(
        PixelCNNPPTrainArgs(
            dataset=args.dataset,
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            lr_decay=args.lr_decay,
            nr_resnet=args.nr_resnet,
            nr_filters=args.nr_filters,
            nr_logistic_mix=args.nr_logistic_mix,
            seed=args.seed,
            sample_batch_size=args.sample_batch_size,
        )
    )
    print(f"saved_checkpoint={ckpt}")


if __name__ == "__main__":
    main()
