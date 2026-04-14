from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from arpg.data import build_train_loader, get_dataset_spec
from arpg.model import TinyARTransformer


@dataclass
class TrainArgs:
    dataset: str
    vocab_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    out_checkpoint: str


def _shift_for_ar(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    return x, y


def train_baseline(args: TrainArgs) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_dataset_spec(args.dataset)
    seq_len = spec.channels * spec.image_size * spec.image_size

    loader = build_train_loader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
    )
    model = TinyARTransformer(vocab_size=args.vocab_size, seq_len=seq_len - 1).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for tokens in pbar:
            tokens = tokens.to(device, non_blocking=True)
            x, y = _shift_for_ar(tokens)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=float(loss.item()))

    Path(args.out_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset": args.dataset,
            "vocab_size": args.vocab_size,
            "seq_len": seq_len,
        },
        args.out_checkpoint,
    )
