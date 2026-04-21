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


def _rescale(x: torch.Tensor) -> torch.Tensor:
    return (x - 0.5) * 2.0


def _rescale_inv(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x + 0.5


def _build_loaders(dataset: str, data_dir: str, batch_size: int) -> tuple[DataLoader, DataLoader, tuple[int, int, int]]:
    ds_transforms = transforms.Compose([transforms.ToTensor(), transforms.Lambda(_rescale)])
    kwargs = {"num_workers": 2, "pin_memory": True, "drop_last": True}

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


def train_pixelcnnpp(args: PixelCNNPPTrainArgs) -> str:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, obs = _build_loaders(args.dataset, args.data_dir, args.batch_size)
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
    ckpt_dir = Path(args.save_dir) / "checkpoints"
    img_dir = Path(args.save_dir) / "samples"
    metrics_dir = Path(args.save_dir) / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(args.epochs):
        model.train()
        train_bits_acc = 0.0
        train_items = 0
        train_bar = tqdm(train_loader, desc=f"train {epoch + 1}/{args.epochs}")
        for x, _ in train_bar:
            x = x.to(device, non_blocking=True)
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
                x = x.to(device, non_blocking=True)
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
        print(f"epoch={epoch + 1} train_bpd={epoch_metrics['train_bpd']:.4f} test_bpd={epoch_metrics['test_bpd']:.4f}")

        if (epoch + 1) % max(1, min(10, args.epochs)) == 0 or (epoch + 1) == args.epochs:
            sample_t, latency_ms, throughput = sample_grid(
                model=model,
                obs=obs,
                sample_op=sample_op,
                batch_size=args.sample_batch_size,
                device=device,
            )
            sample_t = _rescale_inv(sample_t).clamp(0.0, 1.0)
            utils.save_image(sample_t, str(img_dir / f"{run_name}_epoch{epoch + 1}.png"), nrow=5, padding=0)
            epoch_metrics["sample_latency_ms"] = latency_ms
            epoch_metrics["sample_throughput_img_s"] = throughput

    ckpt_path = ckpt_dir / f"{run_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset": args.dataset,
            "obs": obs,
            "nr_resnet": args.nr_resnet,
            "nr_filters": args.nr_filters,
            "nr_logistic_mix": args.nr_logistic_mix,
            "history": history,
        },
        ckpt_path,
    )

    with (metrics_dir / f"{run_name}.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

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
    ckpt = torch.load(checkpoint_path, map_location=device)
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
        "note": "FID should be computed in a separate script from generated samples.",
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_json).open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
