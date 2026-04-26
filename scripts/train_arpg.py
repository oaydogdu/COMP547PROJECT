"""CLI: train PixelARPG model."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from KlassikAR.arpg_runner import ARPGTrainArgs, train_arpg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="fashion_mnist",
                   choices=["fashion_mnist", "mnist", "cifar10"])
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--save-dir",   default="results/arpg")
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--d-model",    type=int,   default=128)
    p.add_argument("--n-heads",    type=int,   default=4)
    p.add_argument("--n-layers",   type=int,   default=6)
    p.add_argument("--seed",       type=int,   default=1)
    args = p.parse_args()

    train_arpg(ARPGTrainArgs(
        dataset=args.dataset, data_dir=args.data_dir,
        save_dir=args.save_dir, batch_size=args.batch_size,
        epochs=args.epochs, lr=args.lr,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, seed=args.seed,
    ))


if __name__ == "__main__":
    main()
