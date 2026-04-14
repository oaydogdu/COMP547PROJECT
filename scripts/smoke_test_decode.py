from __future__ import annotations

import torch

from arpg.decode import randomized_parallel_decode, sequential_decode
from arpg.model import TinyARTransformer


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 256
    seq_len = 64
    batch_size = 4
    model = TinyARTransformer(vocab_size=vocab_size, seq_len=seq_len).to(device).eval()

    seq_tokens, seq_stats = sequential_decode(model, batch_size=batch_size, seq_len=seq_len, device=device)
    par_tokens, par_stats = randomized_parallel_decode(
        model,
        batch_size=batch_size,
        seq_len=seq_len,
        block_size=8,
        schedule="random",
        device=device,
    )

    assert seq_tokens.shape == (batch_size, seq_len)
    assert par_tokens.shape == (batch_size, seq_len)
    assert seq_tokens.dtype == torch.long
    assert par_tokens.dtype == torch.long
    print("OK: decode smoke test passed")
    print(f"sequential latency_ms={seq_stats.latency_ms:.2f} throughput={seq_stats.throughput_img_s:.2f}")
    print(f"parallel latency_ms={par_stats.latency_ms:.2f} throughput={par_stats.throughput_img_s:.2f}")


if __name__ == "__main__":
    main()
