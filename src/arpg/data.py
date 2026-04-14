from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class DatasetSpec:
    name: str
    channels: int
    image_size: int


def _quantize_to_tokens(images: torch.Tensor, vocab_size: int) -> torch.Tensor:
    tokens = torch.clamp((images * (vocab_size - 1)).round().long(), 0, vocab_size - 1)
    return tokens


def get_dataset_spec(name: str) -> DatasetSpec:
    if name == "fashion_mnist":
        return DatasetSpec(name=name, channels=1, image_size=28)
    if name == "cifar10":
        return DatasetSpec(name=name, channels=3, image_size=32)
    raise ValueError(f"Unsupported dataset: {name}")


def build_train_loader(
    dataset_name: str,
    batch_size: int,
    vocab_size: int,
    num_workers: int = 2,
) -> DataLoader:
    if dataset_name == "fashion_mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    elif dataset_name == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        tokens = _quantize_to_tokens(images, vocab_size=vocab_size)
        b, c, h, w = tokens.shape
        flat = tokens.view(b, c * h * w)
        return flat

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
