from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from KlassikAR.pixelcnnpp_model import PixelCNNPP
from KlassikAR.pixelcnnpp_utils import (
    discretized_mix_logistic_loss,
    discretized_mix_logistic_loss_1d,
    sample_from_discretized_mix_logistic,
    sample_from_discretized_mix_logistic_1d,
)
from common.checkpointing import load_training_checkpoint, save_training_checkpoint, write_metrics


@dataclass
class PixelCNNPPTrainArgs:
    dataset: str
    data_dir: str
    save_dir: str
    batch_size: int
    epochs: int
    lr: float
    lr_decay: float
    nr_resnet: int
    nr_filters: int
    nr_logistic_mix: int
    seed: int
    sample_batch_size: int
    num_workers: int = 0
    save_every_epochs: int = 5
    resume_from: str | None = None
    auto_resume: bool = True


def _rescale(x: torch.Tensor) -> torch.Tensor:
    return (x - 0.5) * 2.0


def _rescale_inv(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x + 0.5


def _build_loaders(
    dataset: str, data_dir: str, batch_size: int, num_workers: int = 0
) -> tuple[DataLoader, DataLoader, tuple[int, int, int]]:
    ds_transforms = transforms.Compose([transforms.ToTensor(), transforms.Lambda(_rescale)])
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": num_workers > 0 and torch.cuda.is_available(),
        "drop_last": True,
    }

    if dataset == "fashion_mnist":
        train_set = datasets.FashionMNIST(data_dir, download=True, train=True, transform=ds_transforms)
        test_set = datasets.FashionMNIST(data_dir, train=False, transform=ds_transforms)
        obs = (1, 28, 28)
    elif dataset == "mnist":
        train_set = datasets.MNIST(data_dir, download=True, train=True, transform=ds_transforms)
        test_set = datasets.MNIST(data_dir, train=False, transform=ds_transforms)
        obs = (1, 28, 28)
    elif dataset == "cifar10":
        train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=ds_transforms)
        test_set = datasets.CIFAR10(data_dir, train=False, transform=ds_transforms)
        obs = (3, 32, 32)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader, obs


def _loss_and_sample_ops(input_channels: int, nr_logistic_mix: int):
    if input_channels == 1:
        return (
            lambda real, pred: discretized_mix_logistic_loss_1d(real, pred),
            lambda pred: sample_from_discretized_mix_logistic_1d(pred, nr_logistic_mix),
        )
    return (
        lambda real, pred: discretized_mix_logistic_loss(real, pred),
        lambda pred: sample_from_discretized_mix_logistic(pred, nr_logistic_mix),
    )


@torch.no_grad()
def sample_grid(
    model: PixelCNNPP,
    obs: tuple[int, int, int],
    sample_op,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    model.eval()
    data = torch.zeros(batch_size, obs[0], obs[1], obs[2], device=device)
    t0 = time.perf_counter()
    for i in range(obs[1]):
        for j in range(obs[2]):
            out = model(data, sample=True)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample[:, :, i, j]
    elapsed = time.perf_counter() - t0
    latency_ms = (elapsed / batch_size) * 1000.0
    throughput = batch_size / elapsed
    return data, latency_ms, throughput


def _configure_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _checkpoint_extra(args: PixelCNNPPTrainArgs, obs: tuple[int, int, int]) -> dict:
    return {
        "dataset": args.dataset,
        "obs": obs,
        "nr_resnet": args.nr_resnet,
        "nr_filters": args.nr_filters,
        "nr_logistic_mix": args.nr_logistic_mix,
        "train_args": {
            "dataset": args.dataset,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "lr_decay": args.lr_decay,
            "nr_resnet": args.nr_resnet,
            "nr_filters": args.nr_filters,
            "nr_logistic_mix": args.nr_logistic_mix,
            "seed": args.seed,
        },
    }


def find_resume_checkpoint(save_dir: str | Path) -> Path | None:
    ckpt_dir = Path(save_dir) / "checkpoints"
    for name in ("last.pt", "best.pt"):
        path = ckpt_dir / name
        if path.exists():
            return path
    epoch_ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    return epoch_ckpts[-1] if epoch_ckpts else None


def train_pixelcnnpp(args: PixelCNNPPTrainArgs) -> str:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        _configure_cuda()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, obs = _build_loaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers
    )
    input_channels = obs[0]

    model = PixelCNNPP(
        nr_resnet=args.nr_resnet,
        nr_filters=args.nr_filters,
        nr_logistic_mix=args.nr_logistic_mix,
        input_channels=input_channels,
    ).to(device)

    loss_op, sample_op = _loss_and_sample_ops(input_channels, args.nr_logistic_mix)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    run_name = f"pixelcnnpp_{args.dataset}_lr{args.lr:.5f}_res{args.nr_resnet}_f{args.nr_filters}"
    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    img_dir = save_dir / "samples"
    metrics_dir = save_dir / "metrics"
    for d in (ckpt_dir, img_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{run_name}.json"
    history: list[dict] = []
    start_epoch = 0
    best_test_bpd = float("inf")

    resume_path = args.resume_from or (find_resume_checkpoint(args.save_dir) if args.auto_resume else None)
    if resume_path:
        resume_path = Path(resume_path)
        print(f"resuming_from={resume_path}")
        ckpt = load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        history = ckpt.get("history", [])
        start_epoch = int(ckpt.get("epoch", 0))
        best_test_bpd = float(ckpt.get("best_test_bpd", best_test_bpd))
        print(f"resume_epoch={start_epoch} best_test_bpd={best_test_bpd:.4f}")

    extra = _checkpoint_extra(args, obs)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_bits_acc = 0.0
        train_items = 0
        train_bar = tqdm(train_loader, desc=f"train {epoch + 1}/{args.epochs}")
        for x, _ in train_bar:
            x = x.to(device)
            out = model(x)
            loss = loss_op(x, out)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bits = loss.item() / (x.size(0) * math.prod(obs) * math.log(2.0))
            train_bits_acc += bits * x.size(0)
            train_items += x.size(0)
            train_bar.set_postfix(bpd=f"{bits:.4f}")

        scheduler.step()

        model.eval()
        test_bits_acc = 0.0
        test_items = 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                out = model(x)
                loss = loss_op(x, out)
                bits = loss.item() / (x.size(0) * math.prod(obs) * math.log(2.0))
                test_bits_acc += bits * x.size(0)
                test_items += x.size(0)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_bpd": train_bits_acc / max(train_items, 1),
            "test_bpd": test_bits_acc / max(test_items, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch + 1} train_bpd={epoch_metrics['train_bpd']:.4f} "
            f"test_bpd={epoch_metrics['test_bpd']:.4f}"
        )

        if (epoch + 1) % max(1, min(10, args.epochs)) == 0 or (epoch + 1) == args.epochs:
            sample_t, latency_ms, throughput = sample_grid(
                model=model,
                obs=obs,
                sample_op=sample_op,
                batch_size=args.sample_batch_size,
                device=device,
            )
            sample_t = _rescale_inv(sample_t).clamp(0.0, 1.0)
            utils.save_image(
                sample_t,
                str(img_dir / f"{run_name}_epoch{epoch + 1}.png"),
                nrow=5,
                padding=0,
            )
            epoch_metrics["sample_latency_ms"] = latency_ms
            epoch_metrics["sample_throughput_img_s"] = throughput

        write_metrics(history, metrics_path)

        ckpt_payload_extra = {**extra, "best_test_bpd": best_test_bpd}
        save_training_checkpoint(
            ckpt_dir / "last.pt",
            model=model,
            epoch=epoch + 1,
            history=history,
            extra=ckpt_payload_extra,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
            save_training_checkpoint(
                ckpt_dir / f"epoch_{epoch + 1:03d}.pt",
                model=model,
                epoch=epoch + 1,
                history=history,
                extra=ckpt_payload_extra,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            print(f"saved_epoch_checkpoint=epoch_{epoch + 1:03d}.pt")

        if epoch_metrics["test_bpd"] < best_test_bpd:
            best_test_bpd = epoch_metrics["test_bpd"]
            ckpt_payload_extra["best_test_bpd"] = best_test_bpd
            save_training_checkpoint(
                ckpt_dir / "best.pt",
                model=model,
                epoch=epoch + 1,
                history=history,
                extra=ckpt_payload_extra,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            print(f"saved_best_checkpoint test_bpd={best_test_bpd:.4f}")

    final_extra = {**extra, "best_test_bpd": best_test_bpd}
    ckpt_path = ckpt_dir / f"{run_name}.pt"
    save_training_checkpoint(
        ckpt_path,
        model=model,
        epoch=args.epochs,
        history=history,
        extra=final_extra,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    print(f"saved_checkpoint={ckpt_path}")
    return str(ckpt_path)


@torch.no_grad()
def evaluate_pixelcnnpp_checkpoint(
    checkpoint_path: str,
    out_json: str,
    out_grid: str,
    sample_batch_size: int = 25,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    sample_t, latency_ms, throughput = sample_grid(model, obs, sample_op, sample_batch_size, device)
    sample_t = _rescale_inv(sample_t).clamp(0.0, 1.0)
    Path(out_grid).parent.mkdir(parents=True, exist_ok=True)
    utils.save_image(sample_t, out_grid, nrow=5, padding=0)

    result = {
        "checkpoint": checkpoint_path,
        "dataset": ckpt["dataset"],
        "sample_batch_size": sample_batch_size,
        "latency_ms_per_image": latency_ms,
        "throughput_img_per_s": throughput,
        "best_test_bpd": ckpt.get("best_test_bpd"),
        "epoch": ckpt.get("epoch"),
        "note": "FID should be computed in a separate script from generated samples.",
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_json).open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
