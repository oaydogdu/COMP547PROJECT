"""Checkpoint helpers for long Colab training runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    epoch: int,
    history: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "history": history,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    map_location = device or "cpu"
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


def write_metrics(history: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def backup_results_tree(src: str | Path, drive_root: str | Path, tag: str) -> Path:
    """Copy a results folder to Google Drive with a timestamped tag."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"Nothing to back up: {src}")
    dest = Path(drive_root) / tag / src.name
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    return dest


def snapshot_to_drive(src: str | Path, drive_dest: str | Path) -> Path:
    """Overwrite a stable Drive folder (used for resume after disconnect)."""
    src = Path(src)
    dest = Path(drive_dest)
    if not src.exists():
        raise FileNotFoundError(f"Nothing to back up: {src}")
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    return dest


def find_arpg_resume_source(drive_base: str | Path) -> Path | None:
    """Pick the best ARPG backup on Drive: stable latest, then newest timestamped."""
    base = Path(drive_base)
    stable = base / "arpg_fashion_latest"
    if (stable / "checkpoints" / "last.pt").exists():
        return stable
    if (stable / "checkpoints").exists() and list((stable / "checkpoints").glob("*.pt")):
        return stable

    candidates: list[Path] = []
    for p in base.glob("arpg_*"):
        folder = p / "arpg_fashion" if (p / "arpg_fashion").exists() else p
        if (folder / "checkpoints").exists() and list((folder / "checkpoints").glob("*.pt")):
            candidates.append(folder)
    direct = base / "arpg_fashion"
    if (direct / "checkpoints").exists() and list((direct / "checkpoints").glob("*.pt")):
        candidates.append(direct)
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def restore_arpg_for_resume(
    local_dir: str | Path,
    drive_base: str | Path,
    *,
    force: bool = False,
) -> str:
    """
    Restore ARPG checkpoints from Drive when local state is missing.

    Returns status: 'local_ok', 'restored', or 'fresh'.
    """
    local = Path(local_dir)
    ckpt_dir = local / "checkpoints"
    has_local = (ckpt_dir / "last.pt").exists() or bool(list(ckpt_dir.glob("epoch_*.pt")))

    if has_local and not force:
        epoch = "?"
        if (ckpt_dir / "last.pt").exists():
            import torch

            ckpt = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
            epoch = str(ckpt.get("epoch", "?"))
        print(f"local_ok: ARPG checkpoints mevcut (epoch={epoch})")
        return "local_ok"

    src = find_arpg_resume_source(drive_base)
    if src is None:
        print("fresh: Drive'da ARPG yedegi yok, sifirdan baslanacak")
        return "fresh"

    if local.exists():
        shutil.rmtree(local)
    shutil.copytree(src, local)
    import torch

    ckpt = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
    print(f"restored: {src} -> epoch={ckpt.get('epoch', '?')}")
    return "restored"
