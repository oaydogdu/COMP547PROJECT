"""
VQ-VAE for CIFAR-10 tokenization.

Architecture:
    Encoder : 32x32x3  -> 8x8x256  (two stride-2 convolutions)
    Quantizer: 8x8x256 -> 8x8 indices in [0, codebook_size)
    Decoder : 8x8x256  -> 32x32x3  (two stride-2 transposed convolutions)

Usage:
    model = VQVAE()
    x_recon, indices, loss = model(x)     # training
    indices = model.encode(x)             # (B, 8, 8)
    x_recon = model.decode_indices(idx)   # (B, 3, 32, 32)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, n_embeddings: int = 512, emb_dim: int = 256, beta: float = 0.25):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.emb_dim      = emb_dim
        self.beta         = beta
        self.embedding = nn.Embedding(n_embeddings, emb_dim)
        self.embedding.weight.data.uniform_(-1 / n_embeddings, 1 / n_embeddings)

    def forward(self, z: torch.Tensor):
        """z: (B, emb_dim, H, W) -> z_q, indices (B,H,W), vq_loss"""
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)          # (B*H*W, C)

        # L2 distances to all codebook vectors
        d = (z_flat ** 2).sum(1, keepdim=True) \
            + (self.embedding.weight ** 2).sum(1) \
            - 2 * (z_flat @ self.embedding.weight.T)

        indices = d.argmin(1)                                    # (B*H*W,)
        z_q = self.embedding(indices).reshape(B, H, W, C).permute(0, 3, 1, 2)

        vq_loss = self.beta * F.mse_loss(z_q.detach(), z) \
                + F.mse_loss(z_q, z.detach())

        z_q = z + (z_q - z).detach()                            # straight-through
        return z_q, indices.reshape(B, H, W), vq_loss

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: (B, H, W) -> (B, emb_dim, H, W)"""
        B, H, W = indices.shape
        z_q = self.embedding(indices.reshape(-1))
        return z_q.reshape(B, H, W, self.emb_dim).permute(0, 3, 1, 2)


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 3, hidden: int = 128, latent: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,   hidden,   4, stride=2, padding=1),  # 32->16
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden,  hidden*2, 4, stride=2, padding=1),  # 16->8
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden*2, latent,  3, stride=1, padding=1),  # 8->8
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent: int = 256, hidden: int = 128, out_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent,   hidden*2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden*2, hidden, 4, stride=2, padding=1),  # 8->16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden, out_ch,   4, stride=2, padding=1),  # 16->32
            nn.Sigmoid(),
        )

    def forward(self, z_q):
        return self.net(z_q)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels:   int = 3,
        hidden_dim:    int = 128,
        latent_dim:    int = 256,
        n_embeddings:  int = 512,
        beta:          float = 0.25,
    ):
        super().__init__()
        self.encoder   = Encoder(in_channels, hidden_dim, latent_dim)
        self.quantizer = VectorQuantizer(n_embeddings, latent_dim, beta)
        self.decoder   = Decoder(latent_dim, hidden_dim, in_channels)
        self.n_embeddings = n_embeddings
        self.latent_dim   = latent_dim

    def forward(self, x):
        z          = self.encoder(x)
        z_q, idx, vq_loss = self.quantizer(z)
        x_recon    = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        return x_recon, idx, recon_loss + vq_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,32,32) -> indices (B,8,8)"""
        z = self.encoder(x)
        _, idx, _ = self.quantizer(z)
        return idx

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: (B,8,8) -> images (B,3,32,32) in [0,1]"""
        z_q = self.quantizer.lookup(indices)
        return self.decoder(z_q)
