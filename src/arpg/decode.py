from __future__ import annotations

import random
import time
from dataclasses import dataclass

import torch


@dataclass
class DecodeStats:
    latency_ms: float
    throughput_img_s: float


def _next_position_indices(remaining: list[int], block_size: int, schedule: str) -> list[int]:
    if schedule == "random":
        random.shuffle(remaining)
        return remaining[:block_size]
    if schedule == "block-raster":
        return remaining[:block_size]
    raise ValueError(f"Unsupported schedule: {schedule}")


@torch.no_grad()
def sequential_decode(model, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, DecodeStats]:
    start = time.perf_counter()
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    for pos in range(seq_len):
        logits = model(tokens)
        next_tok = logits[:, pos, :].argmax(dim=-1)
        tokens[:, pos] = next_tok
    elapsed = time.perf_counter() - start
    stats = DecodeStats(latency_ms=elapsed * 1000.0, throughput_img_s=batch_size / max(elapsed, 1e-8))
    return tokens, stats


@torch.no_grad()
def randomized_parallel_decode(
    model,
    batch_size: int,
    seq_len: int,
    block_size: int,
    schedule: str,
    device: torch.device,
) -> tuple[torch.Tensor, DecodeStats]:
    start = time.perf_counter()
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    remaining = list(range(seq_len))
    while remaining:
        current = _next_position_indices(remaining, block_size=block_size, schedule=schedule)
        logits = model(tokens)
        for pos in current:
            tokens[:, pos] = logits[:, pos, :].argmax(dim=-1)
            remaining.remove(pos)
    elapsed = time.perf_counter() - start
    stats = DecodeStats(latency_ms=elapsed * 1000.0, throughput_img_s=batch_size / max(elapsed, 1e-8))
    return tokens, stats
