"""
V7 PatchTST Training — Binary Classification (Surge Detection)

Changed from regression (MSE on future close) to classification (BCE on surge label).
The model directly learns to predict surge probability.

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
from tqdm import tqdm


class SurgeDataset(Dataset):
    def __init__(self, npz_path, context_length=60):
        data = np.load(npz_path)
        samples = data["samples"].astype(np.float32)  # (N, 65, 3)
        # Replace NaN with 0, clip extreme values
        samples = np.nan_to_num(samples, nan=0.0, posinf=5.0, neginf=-5.0)
        samples = np.clip(samples, -10.0, 10.0)
        self.samples = samples
        self.labels = data["labels"].astype(np.float32)  # (N,) float for BCE
        self.context_length = context_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        context = sample[:self.context_length]   # (60, 3)
        label = self.labels[idx]
        return (
            torch.from_numpy(context),   # (60, 3)
            torch.tensor(label, dtype=torch.float32),
        )


class PatchTSTClassifier(nn.Module):
    """PatchTST for binary surge classification."""

    def __init__(self, input_channels=3, context_length=60,
                 patch_length=8, stride=4, d_model=128, n_heads=4, n_layers=3,
                 dropout=0.2):
        super().__init__()
        self.input_channels = input_channels
        self.context_length = context_length
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

        # Channel mixing + classification head
        self.channel_mix = nn.Linear(input_channels * d_model, d_model)
        self.head = nn.Linear(d_model, 1)  # single logit output
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, context_length, input_channels)
        Returns:
            logit: (batch,) — raw logit for surge probability
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

        logit = self.head(mixed).squeeze(-1)  # (B,)
        return logit


# Keep old name as alias for backward compatibility with inference
PatchTSTModel = PatchTSTClassifier


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
    print("=" * 60)
    print("  V7 PatchTST Training (Binary Classification)")
    print("=" * 60)
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM   : {vram:.1f} GB")

    # Load data
    train_ds = SurgeDataset(processed_dir / "train.npz", data_cfg["context_length"])
    val_ds = SurgeDataset(processed_dir / "val.npz", data_cfg["context_length"])

    train_loader = DataLoader(train_ds, batch_size=p["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=p["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    surge_rate = float(train_ds.labels.mean())
    print(f"  Train  : {len(train_ds):,} samples (surge rate: {surge_rate:.1%})")
    print(f"  Val    : {len(val_ds):,} samples (surge rate: {val_ds.labels.mean():.1%})")
    print("=" * 60)

    # Model
    model = PatchTSTClassifier(
        input_channels=p["input_channels"],
        context_length=p["context_length"],
        patch_length=p["patch_length"],
        stride=p["stride"],
        d_model=p["d_model"],
        n_heads=p["n_heads"],
        n_layers=p["n_layers"],
        dropout=p["dropout"],
    ).to(device)

    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"  Params : {n_params:,}")
    print(f"  Epochs : {p['epochs']} (early stop patience={p['early_stopping_patience']})")

    # Class weight for imbalanced data
    pos_weight = torch.tensor([(1 - surge_rate) / max(surge_rate, 1e-6)]).to(device)
    print(f"  BCE pos_weight: {pos_weight.item():.2f}")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"],
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=p["epochs"]
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, p["epochs"] + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{p['epochs']} [Train]",
                         leave=False, unit="batch")
        for context, label in train_bar:
            context = context.to(device)
            label = label.to(device)

            logit = model(context)
            loss = criterion(logit, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_loss = loss.item()
            train_loss += batch_loss * context.size(0)
            preds = (torch.sigmoid(logit) >= 0.5).float()
            train_correct += (preds == label).sum().item()
            train_total += label.size(0)
            train_bar.set_postfix(loss=f"{batch_loss:.4f}")

        train_loss /= len(train_ds)
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = 0
        val_fp = 0
        val_fn = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{p['epochs']} [Val]  ",
                       leave=False, unit="batch")
        with torch.no_grad():
            for context, label in val_bar:
                context = context.to(device)
                label = label.to(device)
                logit = model(context)
                loss = criterion(logit, label)
                val_loss += loss.item() * context.size(0)

                preds = (torch.sigmoid(logit) >= 0.5).float()
                val_correct += (preds == label).sum().item()
                val_total += label.size(0)
                val_tp += ((preds == 1) & (label == 1)).sum().item()
                val_fp += ((preds == 1) & (label == 0)).sum().item()
                val_fn += ((preds == 0) & (label == 1)).sum().item()
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss /= len(val_ds)
        val_acc = val_correct / val_total
        val_precision = val_tp / max(val_tp + val_fp, 1)
        val_recall = val_tp / max(val_tp + val_fn, 1)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        flag = " *" if val_loss < best_val_loss else f"  (patience {patience_counter+1}/{p['early_stopping_patience']})"
        print(f"Epoch {epoch:02d}/{p['epochs']} | "
              f"Train {train_loss:.4f} acc={train_acc:.1%} | "
              f"Val {val_loss:.4f} acc={val_acc:.1%} prec={val_precision:.1%} rec={val_recall:.1%} | "
              f"LR {lr_now:.2e} | {elapsed:.0f}s{flag}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "val_precision": val_precision, "val_recall": val_recall,
            "val_tp": val_tp, "val_fp": val_fp, "val_fn": val_fn,
            "lr": lr_now, "elapsed": elapsed
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": p,
                "model_type": "classifier",
                "epoch": epoch,
                "val_loss": val_loss,
                "val_precision": val_precision,
                "val_recall": val_recall,
            }, model_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= p["early_stopping_patience"]:
                print(f"\n  Early stopping triggered at epoch {epoch}.")
                break

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": p,
        "model_type": "classifier",
        "epoch": epoch,
        "val_loss": val_loss,
    }, model_dir / "final_model.pt")

    with open(model_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    total_time = sum(h["elapsed"] for h in history)
    best_epoch = min(history, key=lambda h: h["val_loss"])

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time    : {total_time/60:.1f} min ({total_time:.0f}s)")
    print(f"  Epochs ran    : {len(history)}/{p['epochs']}")
    print(f"  Best epoch    : {best_epoch['epoch']} (val_loss={best_epoch['val_loss']:.4f})")
    print(f"  Best precision: {best_epoch['val_precision']:.2%}")
    print(f"  Best recall   : {best_epoch['val_recall']:.2%}")
    print(f"  Model saved   : {model_dir / 'best_model.pt'}")
    print("=" * 60)

    # Quick sanity check
    best_ckpt = torch.load(model_dir / "best_model.pt", map_location=device,
                           weights_only=False)
    print(f"\n  [Sanity] Best model checkpoint: epoch={best_ckpt['epoch']}, "
          f"val_loss={best_ckpt['val_loss']:.6f}, type={best_ckpt.get('model_type', 'regression')}")


if __name__ == "__main__":
    train()
