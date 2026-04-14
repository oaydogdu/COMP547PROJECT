from __future__ import annotations

import torch
import torch.nn as nn


class TinyARTransformer(nn.Module):
    """
    Small causal transformer for discrete image-token modeling.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size + 1, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones(length, length, device=device, dtype=torch.bool).triu(1)
        return mask

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(tokens) + self.pos_emb[:, : tokens.size(1), :]
        mask = self._causal_mask(tokens.size(1), device=tokens.device)
        x = self.encoder(x, mask=mask)
        x = self.norm(x)
        return self.head(x)
