from __future__ import annotations

import argparse

from KlassikAR.pixelcnnpp_runner import PixelCNNPPTrainArgs, train_pixelcnnpp


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
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-every-epochs", type=int, default=5)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints and train from scratch")
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
            num_workers=args.num_workers,
            save_every_epochs=args.save_every_epochs,
            resume_from=args.resume_from,
            auto_resume=not args.fresh,
        )
    )
    print(f"saved_checkpoint={ckpt}")


if __name__ == "__main__":
    main()
