"""
V7 PatchTST Training — CPU/GPU Auto-Select, Temporal Split

Usage:
  python stock_prediction_v7/src/train/train_v7.py
  (runs on GPU if CUDA available, otherwise CPU)
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml
import json
import time


class SurgeDataset(Dataset):
    def __init__(self, npz_path, context_length=60):
        data = np.load(npz_path)
        self.samples = data["samples"].astype(np.float32)  # (N, 65, 3)
        self.labels = data["labels"].astype(np.int64)       # (N,)
        self.context_length = context_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        context = sample[:self.context_length]   # (60, 3)
        future = sample[self.context_length:]    # (5, 3)
        label = self.labels[idx]
        return (
            torch.from_numpy(context),   # (60, 3)
            torch.from_numpy(future),    # (5, 3)
            torch.tensor(label, dtype=torch.long),
        )


class PatchTSTModel(nn.Module):
    """Simplified PatchTST for surge prediction."""

    def __init__(self, input_channels=3, context_length=60, prediction_length=5,
                 patch_length=8, stride=4, d_model=128, n_heads=4, n_layers=3,
                 dropout=0.2):
        super().__init__()
        self.input_channels = input_channels
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_length = patch_length
        self.stride = stride
        self.d_model = d_model

        # Number of patches
        self.n_patches = (context_length - patch_length) // stride + 1

        # Per-channel patch embedding (channel independence)
        self.patch_embed = nn.Linear(patch_length, d_model)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Channel mixing + output head
        self.channel_mix = nn.Linear(input_channels * d_model, d_model)
        self.head = nn.Linear(d_model, prediction_length)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, context_length, input_channels)
        Returns:
            forecast: (batch, prediction_length) — predicted close price (channel 0)
        """
        B, L, C = x.shape

        # Create patches per channel (channel independence)
        channel_outputs = []
        for c in range(C):
            ch = x[:, :, c]  # (B, L)
            patches = ch.unfold(1, self.patch_length, self.stride)  # (B, n_patches, patch_len)
            patches = self.patch_embed(patches)  # (B, n_patches, d_model)
            patches = patches + self.pos_embed
            out = self.transformer(patches)  # (B, n_patches, d_model)
            out = out.mean(dim=1)  # (B, d_model) — global average pooling
            channel_outputs.append(out)

        # Mix channels
        mixed = torch.cat(channel_outputs, dim=-1)  # (B, C * d_model)
        mixed = self.channel_mix(mixed)  # (B, d_model)
        mixed = self.dropout(torch.relu(mixed))

        forecast = self.head(mixed)  # (B, prediction_length)
        return forecast


def get_device(config_device="auto"):
    if config_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(config_device)


def train(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "v7_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    p = cfg["patchtst"]
    data_cfg = cfg["data"]
    processed_dir = Path(__file__).parent.parent.parent / data_cfg["processed_dir"]
    model_dir = Path(__file__).parent.parent.parent / "models" / "patchtst"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(p["device"])
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    train_ds = SurgeDataset(processed_dir / "train.npz", data_cfg["context_length"])
    val_ds = SurgeDataset(processed_dir / "val.npz", data_cfg["context_length"])

    train_loader = DataLoader(train_ds, batch_size=p["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=p["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Model
    model = PatchTSTModel(
        input_channels=p["input_channels"],
        context_length=p["context_length"],
        prediction_length=p["prediction_length"],
        patch_length=p["patch_length"],
        stride=p["stride"],
        d_model=p["d_model"],
        n_heads=p["n_heads"],
        n_layers=p["n_layers"],
        dropout=p["dropout"],
    ).to(device)

    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"],
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=p["epochs"]
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, p["epochs"] + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for context, future, label in train_loader:
            context = context.to(device)
            # Target: close price channel (index 0) of future
            target = future[:, :, 0].to(device)  # (B, 5)

            pred = model(context)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * context.size(0)

        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for context, future, label in val_loader:
                context = context.to(device)
                target = future[:, :, 0].to(device)
                pred = model(context)
                loss = criterion(pred, target)
                val_loss += loss.item() * context.size(0)
        val_loss /= len(val_ds)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{p['epochs']} | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": p,
                "epoch": epoch,
                "val_loss": val_loss,
            }, model_dir / "best_model.pt")
            print(f"  -> Best model saved (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= p["early_stopping_patience"]:
                print(f"  -> Early stopping at epoch {epoch}")
                break

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": p,
        "epoch": epoch,
        "val_loss": val_loss,
    }, model_dir / "final_model.pt")

    with open(model_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    train()
