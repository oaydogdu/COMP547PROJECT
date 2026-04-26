"""
Training + evaluation pipeline for PixelARPG.

Key functions
-------------
train_arpg        -- trains the masked pixel prediction model.
arpg_decode       -- ARPG-style parallel decode with a given schedule.
run_arpg_sweep    -- sweeps K values x schedules; saves grids + JSON.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torchvision import utils as tvutils
from torch.utils.data import DataLoader
from tqdm import tqdm

from KlassikAR.arpg_model import MASK_ID, PixelARPG


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ARPGTrainArgs:
    dataset: str = "fashion_mnist"
    data_dir: str = "data"
    save_dir: str = "results/arpg"
    batch_size: int = 256
    epochs: int = 20
    lr: float = 3e-4
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 6
    n_levels: int = 256
    dropout: float = 0.1
    seed: int = 1


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _build_loaders(dataset, data_dir, batch_size):
    tf = transforms.ToTensor()
    if dataset == "fashion_mnist":
        train_ds = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=tf)
        test_ds  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
        H, W = 28, 28
    elif dataset == "mnist":
        train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=tf)
        test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
        H, W = 28, 28
    elif dataset == "cifar10":
        train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=tf)
        test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
        H, W = 32, 32
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    kw = dict(num_workers=2, pin_memory=True, drop_last=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
        H, W,
    )


def _to_tokens(x: torch.Tensor, n_levels: int = 256) -> torch.Tensor:
    """[0,1] float image -> (B, H*W) long tokens in [0, n_levels-1]."""
    return (x.flatten(1) * (n_levels - 1)).long().clamp(0, n_levels - 1)


def _random_mask(tokens: torch.Tensor, mask_rate: float):
    """Randomly mask ``mask_rate`` fraction of pixels with MASK_ID.

    Returns (masked_tokens, bool_mask) where bool_mask=True at masked positions.
    """
    B, N = tokens.shape
    n_mask = max(1, int(N * mask_rate))
    idx   = torch.rand(B, N, device=tokens.device).argsort(dim=1)[:, :n_mask]
    masked = tokens.clone()
    bmask  = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
    masked.scatter_(1, idx, MASK_ID)
    bmask.scatter_(1,  idx, True)
    return masked, bmask


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_arpg(args: ARPGTrainArgs) -> str:
    """Train PixelARPG; return checkpoint path."""
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, H, W = _build_loaders(
        args.dataset, args.data_dir, args.batch_size
    )
    model = PixelARPG(
        H=H, W=W, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, n_levels=args.n_levels, dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    metrics_dir = save_dir / "metrics"
    for d in (ckpt_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    run = f"arpg_{args.dataset}_d{args.d_model}_l{args.n_layers}"
    history = []

    for epoch in range(args.epochs):
        # ---- train ----
        model.train()
        acc, n = 0.0, 0
        bar = tqdm(train_loader, desc=f"train {epoch+1}/{args.epochs}")
        for x, _ in bar:
            x      = x.to(device)
            tokens = _to_tokens(x, args.n_levels)
            # Vary mask rate in [0.1, 0.9] so model learns any conditioning
            rate   = 0.1 + 0.8 * torch.rand(1).item()
            masked, bmask = _random_mask(tokens, rate)
            logits = model(masked)
            loss   = F.cross_entropy(logits[bmask], tokens[bmask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bpd = loss.item() / math.log(2.0)
            acc += bpd * x.size(0); n += x.size(0)
            bar.set_postfix(bpd=f"{bpd:.4f}")
        scheduler.step()
        train_bpd = acc / n

        # ---- eval at fixed 50% mask ----
        model.eval()
        acc_e, n_e = 0.0, 0
        with torch.no_grad():
            for x, _ in test_loader:
                x      = x.to(device)
                tokens = _to_tokens(x, args.n_levels)
                masked, bmask = _random_mask(tokens, 0.5)
                logits = model(masked)
                loss   = F.cross_entropy(logits[bmask], tokens[bmask])
                acc_e += (loss.item() / math.log(2.0)) * x.size(0)
                n_e   += x.size(0)
        test_bpd = acc_e / n_e

        ep = {"epoch": epoch + 1, "train_bpd": train_bpd,
              "test_bpd_50pct_mask": test_bpd,
              "lr": float(optimizer.param_groups[0]["lr"])}
        history.append(ep)
        print(f"epoch={epoch+1}  train_bpd={train_bpd:.4f}  test_bpd={test_bpd:.4f}")

    ckpt_path = ckpt_dir / f"{run}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "H": H, "W": W, "d_model": args.d_model, "n_heads": args.n_heads,
        "n_layers": args.n_layers, "n_levels": args.n_levels,
        "dataset": args.dataset, "history": history,
    }, ckpt_path)
    (metrics_dir / f"{run}.json").write_text(json.dumps(history, indent=2))
    print(f"saved_checkpoint={ckpt_path}")
    return str(ckpt_path)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

Schedule = Literal["random", "raster", "row", "column"]


def _decode_order(H: int, W: int, schedule: Schedule, seed: int = 42) -> torch.Tensor:
    """Return a permutation of pixel indices 0..H*W-1 for the given schedule."""
    N = H * W
    if schedule == "random":
        g = torch.Generator().manual_seed(seed)
        return torch.randperm(N, generator=g)
    elif schedule == "raster":
        return torch.arange(N)
    elif schedule == "row":
        g = torch.Generator().manual_seed(seed)
        return torch.cat([torch.randperm(W, generator=g) + r * W for r in range(H)])
    elif schedule == "column":
        g = torch.Generator().manual_seed(seed)
        return torch.cat([torch.randperm(H, generator=g) * W + c for c in range(W)])
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


@torch.no_grad()
def arpg_decode(
    model: PixelARPG,
    n_samples: int,
    n_steps: int,
    device: torch.device,
    schedule: Schedule = "random",
    seed: int = 42,
) -> tuple[torch.Tensor, float]:
    """
    ARPG-style K-step parallel decode.

    Parameters
    ----------
    n_steps  : K = number of forward passes.
               K=1  -> fully parallel (fastest, lowest quality).
               K=N  -> one pixel/step (sequential, best quality).
    schedule : pixel reveal order -- "random" | "raster" | "row" | "column".

    Returns
    -------
    images  : (B, 1, H, W) float in [0, 1]
    elapsed : wall-clock seconds
    """
    H, W, N = model.H, model.W, model.N
    order = _decode_order(H, W, schedule, seed=seed).to(device)
    base, rem = divmod(N, n_steps)
    sizes = [base + (1 if i < rem else 0) for i in range(n_steps)]

    tokens = torch.full((n_samples, N), MASK_ID, dtype=torch.long, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    cursor = 0
    for step_size in sizes:
        logits  = model(tokens)                            # (B, N, C)
        sampled = torch.multinomial(
            logits.softmax(-1).view(-1, model.n_levels), 1
        ).view(n_samples, N)
        idx = order[cursor:cursor + step_size]
        tokens[:, idx] = sampled[:, idx]
        cursor += step_size

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    imgs = tokens.float().view(n_samples, 1, H, W) / (model.n_levels - 1)
    return imgs, elapsed


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_arpg_sweep(
    checkpoint_path: str,
    out_dir: str,
    k_values: tuple = (1, 2, 4, 7, 14, 28, 56, 112, 196, 392, 784),
    schedules: tuple = ("random", "raster", "row"),
    n_samples: int = 25,
    seed: int = 42,
) -> dict:
    """Sweep K x schedule; save grids + sweep.json."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = PixelARPG(
        H=ckpt["H"], W=ckpt["W"],
        d_model=ckpt["d_model"], n_heads=ckpt["n_heads"],
        n_layers=ckpt["n_layers"], n_levels=ckpt["n_levels"],
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    out_dir   = Path(out_dir)
    grids_dir = out_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    # warm-up
    arpg_decode(model, 1, 2, device, schedule="random", seed=seed)

    results = []
    for sched in schedules:
        for K in k_values:
            imgs, elapsed = arpg_decode(
                model, n_samples, int(K), device, schedule=sched, seed=seed
            )
            latency_ms = (elapsed / n_samples) * 1000.0
            throughput  = n_samples / max(elapsed, 1e-9)
            grid_path   = grids_dir / f"{sched}_K{int(K):04d}.png"
            tvutils.save_image(imgs, str(grid_path), nrow=5, padding=2)

            rec = dict(schedule=sched, K=int(K),
                       latency_ms_per_image=latency_ms,
                       throughput_img_per_s=throughput,
                       grid=str(grid_path))
            results.append(rec)
            print(f"[{sched:8s}] K={int(K):4d}  "
                  f"{latency_ms:8.2f} ms/img  {throughput:.3f} img/s")

    summary = {"checkpoint": checkpoint_path, "dataset": ckpt["dataset"], "sweep": results}
    (out_dir / "sweep.json").write_text(json.dumps(summary, indent=2))
    print(f"saved: {out_dir / 'sweep.json'}")
    return summary
