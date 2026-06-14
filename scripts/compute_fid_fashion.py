"""Generate samples and compute FID for Fashion-MNIST baseline or ARPG."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ARPG.arpg_runner import arpg_decode
from ARPG.arpg_model import PixelARPG
from KlassikAR.pixelcnnpp_model import PixelCNNPP
from KlassikAR.pixelcnnpp_runner import _loss_and_sample_ops, _rescale_inv, sample_grid

REF_NAME = "fashion_mnist_train_clean"
DEFAULT_REF_DIR = Path("data/fashion_mnist_train_ref")


def _to_rgb_batch(imgs: torch.Tensor) -> torch.Tensor:
    if imgs.size(1) == 1:
        return imgs.repeat(1, 3, 1, 1)
    return imgs


def prepare_reference_folder(ref_dir: Path, data_dir: str = "data") -> Path:
    """Export Fashion-MNIST train split to PNG for folder-to-folder FID."""
    ref_dir.mkdir(parents=True, exist_ok=True)
    existing = list(ref_dir.glob("*.png"))
    if len(existing) >= 60000:
        print(f"reference_ready={ref_dir} count={len(existing)}")
        return ref_dir

    ds = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
    for i in range(len(ds)):
        img, _ = ds[i]
        save_image(_to_rgb_batch(img.unsqueeze(0)), ref_dir / f"{i:05d}.png")
        if (i + 1) % 10000 == 0:
            print(f"exported_ref={i + 1}/{len(ds)}")
    print(f"reference_ready={ref_dir} count={len(ds)}")
    return ref_dir


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
        imgs = _to_rgb_batch(imgs)
        for i in range(b):
            save_image(imgs[i], out_dir / f"{made + i:05d}.png")
        made += b
        if made % 100 == 0 or made == n:
            print(f"generated={made}/{n}")


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
        imgs = _to_rgb_batch(imgs)
        for i in range(b):
            save_image(imgs[i], out_dir / f"{made + i:05d}.png")
        made += b
        if made % 100 == 0 or made == n:
            print(f"generated={made}/{n}")


def _compute_fid(gen_dir: Path, ref_dir: Path) -> float:
    from cleanfid import fid as clean_fid

    stats_exist = clean_fid.test_stats_exists(REF_NAME, mode="clean")
    if not stats_exist:
        print(f"building_custom_stats={REF_NAME} (one-time, ~5-10 min)")
        clean_fid.make_custom_stats(REF_NAME, str(ref_dir), mode="clean", num_workers=0)

    score = clean_fid.compute_fid(
        str(gen_dir),
        dataset_name=REF_NAME,
        dataset_split="custom",
        mode="clean",
        num_workers=0,
    )
    return float(score)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "arpg"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ref-dir", default=str(DEFAULT_REF_DIR))
    p.add_argument("--n-samples", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=28)
    p.add_argument("--schedule", default="random")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--compute-fid", action="store_true")
    p.add_argument("--skip-gen", action="store_true", help="Use existing images in out-dir")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    ref_dir = prepare_reference_folder(Path(args.ref_dir))

    if not args.skip_gen:
        if args.model == "baseline":
            _generate_baseline(args.checkpoint, out_dir, args.n_samples, args.batch_size, args.seed)
        else:
            _generate_arpg(args.checkpoint, out_dir, args.n_samples, args.k, args.schedule, args.seed, args.top_p)
    else:
        n_existing = len(list(out_dir.glob("*.png")))
        print(f"skip_gen=True existing_images={n_existing}")

    result = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "n_samples": args.n_samples,
        "gen_dir": str(out_dir),
        "ref_dir": str(ref_dir),
    }
    if args.model == "arpg":
        result["k"] = args.k
        result["schedule"] = args.schedule

    if args.compute_fid:
        score = _compute_fid(out_dir, ref_dir)
        result["fid"] = score
        print(f"FID={score:.4f}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved={args.out_json}")


if __name__ == "__main__":
    main()
