"""Train VQ-VAE on CIFAR-10 and encode all images to discrete tokens."""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from VQVAE.vqvae import VQVAE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--save-dir",   default="results/vqvae_cifar10")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--n-embeddings", type=int, default=512)
    p.add_argument("--latent-dim",   type=int, default=256)
    p.add_argument("--seed",       type=int,   default=1)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.CIFAR10(args.data_dir, train=True,  download=True, transform=tf)
    test_ds  = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = VQVAE(n_embeddings=args.n_embeddings, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss, n = 0.0, 0
        for x, _ in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            x = x.to(device)
            _, _, loss = model(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = total_loss / n

        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                _, _, loss = model(x)
                val_loss += loss.item() * x.size(0)
                n += x.size(0)
        val_loss /= n

        history.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch+1}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    # Save checkpoint
    ckpt_path = save_dir / "vqvae_cifar10.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_embeddings": args.n_embeddings,
        "latent_dim": args.latent_dim,
        "history": history,
    }, ckpt_path)
    print(f"Saved VQ-VAE: {ckpt_path}")

    # Encode all CIFAR-10 images to tokens
    print("Encoding CIFAR-10 to VQ tokens...")
    model.eval()
    train_loader2 = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=2)
    test_loader2  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)

    def encode_all(loader):
        all_idx = []
        with torch.no_grad():
            for x, _ in tqdm(loader, desc="encoding"):
                idx = model.encode(x.to(device))  # (B, 8, 8)
                all_idx.append(idx.cpu())
        return torch.cat(all_idx, dim=0)  # (N, 8, 8)

    train_tokens = encode_all(train_loader2)
    test_tokens  = encode_all(test_loader2)

    token_path = save_dir / "cifar10_tokens.pt"
    torch.save({
        "train": train_tokens,   # (50000, 8, 8)
        "test":  test_tokens,    # (10000, 8, 8)
        "n_embeddings": args.n_embeddings,
        "vqvae_ckpt": str(ckpt_path),
    }, token_path)
    print(f"Saved tokens: {token_path}")
    print(f"  train: {train_tokens.shape}, test: {test_tokens.shape}")
    (save_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
