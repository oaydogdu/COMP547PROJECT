"""
PixelARPG: Bidirectional pixel-level model for randomized parallel decoding.

Architecture:
    - Input  : quantized pixel values (0-255) + mask token (256) at each position
    - Model  : bidirectional Transformer (no causal mask) + 2-D positional embeddings
    - Output : 256-way softmax per pixel position

Training (masked pixel prediction):
    - Each batch: randomly mask r ~ Uniform(0.1, 0.9) fraction of pixels
    - Predict masked pixels from the rest (bidirectional context)
    - Loss: cross-entropy on masked positions only

Why this enables randomized parallel decoding:
    - The model learns p(x_mask | x_obs) for *any* arbitrary subset
    - At inference: reveal pixels in any order, K per step:
          K = 1   -> slowest, best quality (one pixel per forward pass)
          K = N   -> fastest, lowest quality (all pixels in one pass)
    - The decoding ORDER can be random (ARPG-style) or structured
      (row-by-row, raster, etc.) -- enabling the schedule comparison
      described in the proposal.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MASK_ID: int = 256  # special token index for masked / unknown pixels


class PixelARPG(nn.Module):
    """Bidirectional masked pixel prediction model."""

    def __init__(
        self,
        H: int = 28,
        W: int = 28,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        n_levels: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.H = H
        self.W = W
        self.N = H * W
        self.n_levels = n_levels

        # Pixel-value embedding: 0..255 + mask token (index 256)
        self.pixel_embed = nn.Embedding(n_levels + 1, d_model)

        # Separate row / col position embeddings (concatenated -> d_model)
        assert d_model % 2 == 0
        self.row_embed = nn.Embedding(H, d_model // 2)
        self.col_embed = nn.Embedding(W, d_model // 2)

        # Bidirectional Transformer encoder (no causal mask)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Per-position classification head
        self.head = nn.Linear(d_model, n_levels)

    def _pos_embed(self, device: torch.device) -> torch.Tensor:
        """Build (N, d_model) position tensor for all H*W pixels."""
        rows = torch.arange(self.H, device=device).repeat_interleave(self.W)
        cols = torch.arange(self.W, device=device).repeat(self.H)
        return torch.cat([self.row_embed(rows), self.col_embed(cols)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, H*W) Long tensor -- pixel values 0..255;
            masked positions must be set to MASK_ID (= 256).

        Returns
        -------
        logits : (B, H*W, n_levels)
        """
        pos = self._pos_embed(x.device).unsqueeze(0)  # (1, N, d)
        h = self.pixel_embed(x) + pos                 # (B, N, d)
        h = self.transformer(h)
        h = self.norm(h)
        return self.head(h)                            # (B, N, n_levels)
