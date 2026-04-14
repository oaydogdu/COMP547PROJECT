from __future__ import annotations

import json
from pathlib import Path

import torch

from arpg.decode import randomized_parallel_decode, sequential_decode
from arpg.model import TinyARTransformer


def load_model(checkpoint_path: str, device: torch.device) -> tuple[TinyARTransformer, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    seq_len = int(ckpt["seq_len"]) - 1
    model = TinyARTransformer(
        vocab_size=int(ckpt["vocab_size"]),
        seq_len=seq_len,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def evaluate_decode_modes(
    checkpoint_path: str,
    out_json: str,
    batch_size: int = 16,
    schedule: str = "random",
    block_size: int = 16,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(checkpoint_path, device=device)
    seq_len = int(ckpt["seq_len"]) - 1

    _, seq_stats = sequential_decode(model, batch_size=batch_size, seq_len=seq_len, device=device)
    _, par_stats = randomized_parallel_decode(
        model,
        batch_size=batch_size,
        seq_len=seq_len,
        block_size=block_size,
        schedule=schedule,
        device=device,
    )

    result = {
        "checkpoint": checkpoint_path,
        "batch_size": batch_size,
        "schedule": schedule,
        "block_size": block_size,
        "sequential": {
            "latency_ms": seq_stats.latency_ms,
            "throughput_img_s": seq_stats.throughput_img_s,
        },
        "parallel": {
            "latency_ms": par_stats.latency_ms,
            "throughput_img_s": par_stats.throughput_img_s,
        },
        # FID placeholder: filled by dedicated FID script in later phase.
        "fid": None,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_json).open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
