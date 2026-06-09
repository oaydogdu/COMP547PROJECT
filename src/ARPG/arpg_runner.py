"""Training + evaluation pipeline for PixelARPG."""

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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import utils as tvutils
from tqdm import tqdm

from ARPG.arpg_model import MASK_ID, PixelARPG
from common.checkpointing import load_training_checkpoint, save_training_checkpoint, write_metrics


@dataclass
class ARPGTrainArgs:
    dataset: str = "fashion_mnist"
    data_dir: str = "data"
    save_dir: str = "results/arpg"
    batch_size: int = 16
    epochs: int = 20
    lr: float = 3e-4
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 6
    n_levels: int = 256
    dropout: float = 0.05
    seed: int = 1
    num_workers: int = 0
    save_every_epochs: int = 5
    resume_from: str | None = None
    auto_resume: bool = True


def _build_loaders(dataset: str, data_dir: str, batch_size: int, num_workers: int = 0):
    if dataset == "fashion_mnist":
        tf = transforms.ToTensor()
        train_ds = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tf)
        test_ds = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
        H, W, C = 28, 28, 1
    elif dataset == "mnist":
        tf = transforms.ToTensor()
        train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
        test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
        H, W, C = 28, 28, 1
    elif dataset == "cifar10":
        tf = transforms.Compose([transforms.Grayscale(), transforms.Resize(16), transforms.ToTensor()])
        train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
        test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
        H, W, C = 16, 16, 1
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    kw = {
        "num_workers": num_workers,
        "pin_memory": num_workers > 0 and torch.cuda.is_available(),
        "drop_last": True,
    }
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw),
        H, W, C,
    )


def _to_tokens(x: torch.Tensor, n_levels: int = 256) -> torch.Tensor:
    return (x.flatten(1) * (n_levels - 1)).long().clamp(0, n_levels - 1)


def _random_mask(tokens: torch.Tensor, mask_rate: float):
    B, N = tokens.shape
    n_mask = max(1, int(N * mask_rate))
    perm = torch.rand(B, N, device=tokens.device).argsort(dim=1)
    idx = perm[:, N - n_mask:]
    masked = tokens.clone()
    bmask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
    masked.scatter_(1, idx, MASK_ID)
    bmask.scatter_(1, idx, True)
    return masked, bmask


def find_resume_checkpoint(save_dir: str | Path) -> Path | None:
    ckpt_dir = Path(save_dir) / "checkpoints"
    for name in ("last.pt", "best.pt"):
        path = ckpt_dir / name
        if path.exists():
            return path
    epoch_ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    return epoch_ckpts[-1] if epoch_ckpts else None


def _checkpoint_extra(args: ARPGTrainArgs, H: int, W: int, C: int) -> dict:
    return {
        "H": H,
        "W": W,
        "C": C,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "n_levels": args.n_levels,
        "dataset": args.dataset,
    }


def train_arpg(args: ARPGTrainArgs) -> str:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, H, W, C = _build_loaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers
    )
    model = PixelARPG(
        H=H, W=W, C=C, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, n_levels=args.n_levels, dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    use_amp = device.type == "cuda"

    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    metrics_dir = save_dir / "metrics"
    for d in (ckpt_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    run = f"arpg_{args.dataset}_d{args.d_model}_l{args.n_layers}"
    metrics_path = metrics_dir / f"{run}.json"
    history: list[dict] = []
    start_epoch = 0
    best_test_bpd = float("inf")
    extra = _checkpoint_extra(args, H, W, C)

    resume_path = args.resume_from or (find_resume_checkpoint(args.save_dir) if args.auto_resume else None)
    if resume_path:
        resume_path = Path(resume_path)
        print(f"resuming_from={resume_path}")
        ckpt = load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        history = ckpt.get("history", [])
        start_epoch = int(ckpt.get("epoch", 0))
        best_test_bpd = float(ckpt.get("best_test_bpd", best_test_bpd))
        print(f"resume_epoch={start_epoch} best_test_bpd={best_test_bpd:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        acc, n = 0.0, 0
        bar = tqdm(train_loader, desc=f"train {epoch + 1}/{args.epochs}")
        for x, _ in bar:
            x = x.to(device)
            tokens = _to_tokens(x, args.n_levels)
            u = torch.rand(1).item()
            rate = math.cos(u * math.pi / 2)
            masked, bmask = _random_mask(tokens, rate)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(masked)
                loss = F.cross_entropy(logits[bmask], tokens[bmask])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            bpd = loss.item() / math.log(2.0)
            acc += bpd * x.size(0)
            n += x.size(0)
            bar.set_postfix(bpd=f"{bpd:.4f}")
        scheduler.step()
        train_bpd = acc / n

        model.eval()
        acc_e, n_e = 0.0, 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                tokens = _to_tokens(x, args.n_levels)
                masked, bmask = _random_mask(tokens, 0.5)
                logits = model(masked)
                loss = F.cross_entropy(logits[bmask], tokens[bmask])
                acc_e += (loss.item() / math.log(2.0)) * x.size(0)
                n_e += x.size(0)
        test_bpd = acc_e / n_e

        ep = {
            "epoch": epoch + 1,
            "train_bpd": train_bpd,
            "test_bpd_50pct_mask": test_bpd,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(ep)
        print(f"epoch={epoch + 1}  train_bpd={train_bpd:.4f}  test_bpd={test_bpd:.4f}")

        write_metrics(history, metrics_path)
        payload_extra = {**extra, "best_test_bpd": best_test_bpd}
        save_training_checkpoint(
            ckpt_dir / "last.pt",
            model=model,
            epoch=epoch + 1,
            history=history,
            extra=payload_extra,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

        if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
            save_training_checkpoint(
                ckpt_dir / f"epoch_{epoch + 1:03d}.pt",
                model=model,
                epoch=epoch + 1,
                history=history,
                extra=payload_extra,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            print(f"saved_epoch_checkpoint=epoch_{epoch + 1:03d}.pt")

        if test_bpd < best_test_bpd:
            best_test_bpd = test_bpd
            payload_extra["best_test_bpd"] = best_test_bpd
            save_training_checkpoint(
                ckpt_dir / "best.pt",
                model=model,
                epoch=epoch + 1,
                history=history,
                extra=payload_extra,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            print(f"saved_best_checkpoint test_bpd={best_test_bpd:.4f}")

    payload_extra = {**extra, "best_test_bpd": best_test_bpd}
    ckpt_path = ckpt_dir / f"{run}.pt"
    save_training_checkpoint(
        ckpt_path,
        model=model,
        epoch=args.epochs,
        history=history,
        extra=payload_extra,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    print(f"saved_checkpoint={ckpt_path}")
    return str(ckpt_path)


Schedule = Literal["random", "raster", "row", "column"]


def _arccos_sizes(N: int, n_steps: int) -> list[int]:
    counts = [round(N * math.cos(t / n_steps * math.pi / 2)) for t in range(n_steps + 1)]
    counts[0] = N
    counts[-1] = 0
    sizes = [max(0, counts[t] - counts[t + 1]) for t in range(n_steps)]
    remainder = N - sum(sizes)
    if remainder != 0:
        for i in range(len(sizes) - 1, -1, -1):
            if sizes[i] > 0:
                sizes[i] += remainder
                break
    return sizes


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    remove = (cumsum - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    return probs.new_zeros(probs.shape).scatter_(-1, sorted_idx, sorted_probs)


def _decode_order(H: int, W: int, schedule: Schedule, seed: int = 42) -> torch.Tensor:
    N = H * W
    if schedule == "random":
        g = torch.Generator().manual_seed(seed)
        return torch.randperm(N, generator=g)
    if schedule == "raster":
        return torch.arange(N)
    if schedule == "row":
        g = torch.Generator().manual_seed(seed)
        return torch.cat([torch.randperm(W, generator=g) + r * W for r in range(H)])
    if schedule == "column":
        g = torch.Generator().manual_seed(seed)
        return torch.cat([torch.randperm(H, generator=g) * W + c for c in range(W)])
    raise ValueError(f"Unknown schedule: {schedule}")


@torch.no_grad()
def arpg_decode(
    model: PixelARPG,
    n_samples: int,
    n_steps: int,
    device: torch.device,
    schedule: Schedule = "random",
    seed: int = 42,
    top_p: float = 0.9,
    temperature: float = 1.0,
    confidence_guided: bool = False,
) -> tuple[torch.Tensor, float]:
    H, W, N = model.H, model.W, model.N
    order = _decode_order(H, W, schedule, seed=seed).to(device)
    sizes = _arccos_sizes(N, n_steps)

    tokens = torch.full((n_samples, N), MASK_ID, dtype=torch.long, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    cursor = 0
    for step_size in sizes:
        if step_size == 0:
            continue
        logits = model(tokens)
        if temperature != 1.0:
            logits = logits / temperature
        probs = logits.softmax(-1).view(-1, model.n_levels)
        if top_p < 1.0:
            probs = _top_p_filter(probs, top_p)
        sampled = torch.multinomial(probs, 1).view(n_samples, N)

        if confidence_guided:
            sample_probs = (
                probs.view(n_samples, N, model.n_levels)
                .gather(-1, sampled.unsqueeze(-1))
                .squeeze(-1)
            )
            is_masked = tokens == MASK_ID
            sample_probs = sample_probs.masked_fill(~is_masked, float("-inf"))
            k = min(step_size, int(is_masked.sum(dim=1).max().item()))
            if k > 0:
                _, top_idx = sample_probs.topk(k, dim=1)
                tokens.scatter_(1, top_idx, sampled.gather(1, top_idx))
        else:
            idx = order[cursor:cursor + step_size]
            tokens[:, idx] = sampled[:, idx]
            cursor += step_size

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    imgs = tokens.float().view(n_samples, model.C, H, W) / (model.n_levels - 1)
    return imgs, elapsed


def run_arpg_sweep(
    checkpoint_path: str,
    out_dir: str,
    k_values: tuple = (1, 2, 4, 7, 14, 28, 56, 112, 196, 392, 784),
    schedules: tuple = ("random", "raster", "row"),
    n_samples: int = 25,
    seed: int = 42,
    top_p: float = 0.9,
    temperature: float = 1.0,
    confidence_guided: bool = False,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = PixelARPG(
        H=ckpt["H"], W=ckpt["W"], C=ckpt.get("C", 1),
        d_model=ckpt["d_model"], n_heads=ckpt["n_heads"],
        n_layers=ckpt["n_layers"], n_levels=ckpt["n_levels"],
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    out_dir = Path(out_dir)
    grids_dir = out_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    arpg_decode(model, 1, 2, device, schedule="random", seed=seed, confidence_guided=confidence_guided)
    tag = "_cg" if confidence_guided else ""
    results = []

    for sched in schedules:
        for K in k_values:
            imgs, elapsed = arpg_decode(
                model, n_samples, int(K), device, schedule=sched, seed=seed,
                top_p=top_p, temperature=temperature, confidence_guided=confidence_guided,
            )
            latency_ms = (elapsed / n_samples) * 1000.0
            throughput = n_samples / max(elapsed, 1e-9)
            grid_path = grids_dir / f"{sched}{tag}_K{int(K):04d}.png"
            tvutils.save_image(imgs, str(grid_path), nrow=5, padding=2)
            results.append({
                "schedule": sched,
                "K": int(K),
                "confidence_guided": confidence_guided,
                "latency_ms_per_image": latency_ms,
                "throughput_img_per_s": throughput,
                "grid": str(grid_path),
            })
            print(f"[{sched:8s}{tag}] K={int(K):4d}  {latency_ms:8.2f} ms/img  {throughput:.3f} img/s")

    summary = {
        "checkpoint": checkpoint_path,
        "dataset": ckpt["dataset"],
        "confidence_guided": confidence_guided,
        "sweep": results,
    }
    (out_dir / "sweep.json").write_text(json.dumps(summary, indent=2))
    print(f"saved: {out_dir / 'sweep.json'}")
    return summary
