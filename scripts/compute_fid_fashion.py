"""Generate samples and compute FID for Fashion-MNIST baseline or ARPG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from ARPG.arpg_runner import arpg_decode
from ARPG.arpg_model import PixelARPG
from KlassikAR.pixelcnnpp_model import PixelCNNPP
from KlassikAR.pixelcnnpp_runner import _loss_and_sample_ops, _rescale_inv, sample_grid


def _generate_baseline(checkpoint: str, out_dir: Path, n: int, batch_size: int, seed: int) -> None:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    obs = tuple(ckpt["obs"])
    model = PixelCNNPP(
        nr_resnet=int(ckpt["nr_resnet"]),
        nr_filters=int(ckpt["nr_filters"]),
        nr_logistic_mix=int(ckpt["nr_logistic_mix"]),
        input_channels=int(obs[0]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _, sample_op = _loss_and_sample_ops(int(obs[0]), int(ckpt["nr_logistic_mix"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    while made < n:
        b = min(batch_size, n - made)
        imgs, _, _ = sample_grid(model, obs, sample_op, b, device)
        imgs = _rescale_inv(imgs).clamp(0.0, 1.0)
        for i in range(b):
            save_image(imgs[i], out_dir / f"{made + i:05d}.png")
        made += b


def _generate_arpg(
    checkpoint: str,
    out_dir: Path,
    n: int,
    k: int,
    schedule: str,
    seed: int,
    top_p: float,
) -> None:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = PixelARPG(
        H=ckpt["H"], W=ckpt["W"], C=ckpt.get("C", 1),
        d_model=ckpt["d_model"], n_heads=ckpt["n_heads"],
        n_layers=ckpt["n_layers"], n_levels=ckpt["n_levels"],
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    while made < n:
        b = min(25, n - made)
        imgs, _ = arpg_decode(model, b, k, device, schedule=schedule, seed=seed + made, top_p=top_p)
        for i in range(b):
            save_image(imgs[i], out_dir / f"{made + i:05d}.png")
        made += b


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "arpg"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-samples", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=28)
    p.add_argument("--schedule", default="random")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--compute-fid", action="store_true")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if args.model == "baseline":
        _generate_baseline(args.checkpoint, out_dir, args.n_samples, args.batch_size, args.seed)
    else:
        _generate_arpg(args.checkpoint, out_dir, args.n_samples, args.k, args.schedule, args.seed, args.top_p)

    result = {"model": args.model, "checkpoint": args.checkpoint, "n_samples": args.n_samples, "gen_dir": str(out_dir)}
    if args.compute_fid:
        from cleanfid import fid as clean_fid

        score = clean_fid.compute_fid(
            str(out_dir),
            dataset_name="FashionMNIST",
            dataset_res=28,
            dataset_split="train",
            mode="clean",
        )
        result["fid"] = float(score)
        print(f"FID={score:.4f}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved={args.out_json}")


if __name__ == "__main__":
    main()
