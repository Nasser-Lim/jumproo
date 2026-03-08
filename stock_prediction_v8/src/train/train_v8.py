# -*- coding: utf-8 -*-
"""
V8 PatchTST Training — Regression (max 5-day return) + Sector Embedding + Market Context

Key changes from V7:
  - Regression target (Huber loss) instead of BCE classification
  - 8 input channels (was 3): +bollinger, +MACD, +KOSPI/KOSDAQ returns, +market vol
  - Sector embedding (nn.Embedding) concatenated after transformer
  - Surge-weighted loss: samples with return >= 15% get 3x weight
  - No class balancing needed (regression uses all samples)

Usage:
  python stock_prediction_v8/src/train/train_v8.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')


class SurgeRegressionDataset(Dataset):
    """Dataset for regression: predicts max 5-day return."""

    def __init__(self, npz_path, context_length=60):
        data = np.load(npz_path)
        samples = data["samples"].astype(np.float32)  # (N, 60, 8)
        samples = np.nan_to_num(samples, nan=0.0, posinf=5.0, neginf=-5.0)
        samples = np.clip(samples, -10.0, 10.0)
        self.samples = samples
        self.targets = data["targets"].astype(np.float32)  # (N,)
        self.sectors = data["sectors"].astype(np.int64)     # (N,)
        self.context_length = context_length

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.samples[idx]),          # (60, 8)
            torch.tensor(self.sectors[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


class PatchTSTRegressor(nn.Module):
    """PatchTST for regression: predicts max 5-day return.

    Architecture:
      1. Per-channel patching + positional encoding
      2. Transformer encoder (channel-independent)
      3. Channel mixing
      4. Sector embedding concatenation
      5. Regression head → single scalar output
    """

    def __init__(self, ts_channels=8, context_length=60,
                 patch_length=10, stride=5, d_model=128, n_heads=4, n_layers=4,
                 dropout=0.25, num_sectors=170, sector_embed_dim=16):
        super().__init__()
        self.ts_channels = ts_channels
        self.context_length = context_length
        self.patch_length = patch_length
        self.stride = stride
        self.d_model = d_model

        self.n_patches = (context_length - patch_length) // stride + 1

        # Per-channel patch embedding
        self.patch_embed = nn.Linear(patch_length, d_model)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Channel mixing
        self.channel_mix = nn.Sequential(
            nn.Linear(ts_channels * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Sector embedding
        self.sector_embed = nn.Embedding(num_sectors + 1, sector_embed_dim)  # +1 for unknown

        # Regression head
        head_dim = d_model + sector_embed_dim
        self.head = nn.Sequential(
            nn.Linear(head_dim, head_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim // 2, 1),
        )

    def forward(self, x, sector_id):
        """
        Args:
            x: (batch, context_length, ts_channels)
            sector_id: (batch,) int — sector index for embedding

        Returns:
            pred: (batch,) — predicted max 5-day return
        """
        B, L, C = x.shape

        # Per-channel transformer (channel independence)
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

        # Sector embedding
        sec_emb = self.sector_embed(sector_id)  # (B, sector_embed_dim)

        # Concatenate and predict
        combined = torch.cat([mixed, sec_emb], dim=-1)  # (B, d_model + sector_embed_dim)
        pred = self.head(combined).squeeze(-1)  # (B,)

        return pred


def get_device(config_device="auto"):
    if config_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(config_device)


def train(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "v8_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    p = cfg["patchtst"]
    data_cfg = cfg["data"]
    root = Path(__file__).parent.parent.parent
    processed_dir = root / data_cfg["processed_dir"]
    model_dir = root / "models" / "patchtst"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(p["device"])

    print("=" * 60)
    print("  V8 PatchTST Training (Regression + Sector Embedding)")
    print("=" * 60)
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM   : {vram:.1f} GB")

    # Load data
    train_ds = SurgeRegressionDataset(processed_dir / "train.npz", data_cfg["context_length"])
    val_ds = SurgeRegressionDataset(processed_dir / "val.npz", data_cfg["context_length"])

    print(f"  Train  : {len(train_ds):,} samples", flush=True)
    print(f"  Val    : {len(val_ds):,} samples", flush=True)

    # Target stats
    train_targets = train_ds.targets
    surge_mask = train_targets >= data_cfg["surge_threshold"]
    print(f"  Train surge rate: {surge_mask.mean():.2%} ({surge_mask.sum():,} samples)")
    print(f"  Train target: mean={train_targets.mean():.4f}, std={train_targets.std():.4f}, "
          f"max={train_targets.max():.4f}")

    # Compute sample weights for surge-weighted loss
    surge_weight = p.get("surge_weight", 3.0)
    normal_weight = p.get("normal_weight", 1.0)
    sample_weights = np.where(surge_mask, surge_weight, normal_weight).astype(np.float32)
    sample_weights_tensor = torch.from_numpy(sample_weights)

    # Use WeightedRandomSampler for balanced batches
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights_tensor, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=p["batch_size"], sampler=sampler,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=p["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    # Model
    model = PatchTSTRegressor(
        ts_channels=p["ts_channels"],
        context_length=p["context_length"],
        patch_length=p["patch_length"],
        stride=p["stride"],
        d_model=p["d_model"],
        n_heads=p["n_heads"],
        n_layers=p["n_layers"],
        dropout=p["dropout"],
        num_sectors=p["num_sectors"],
        sector_embed_dim=p["sector_embed_dim"],
    ).to(device)

    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"  Params : {n_params:,}")
    print(f"  Input  : {p['ts_channels']} channels + sector embedding ({p['sector_embed_dim']}d)")
    print(f"  Patches: {model.n_patches} (length={p['patch_length']}, stride={p['stride']})")
    print(f"  Loss   : {p['loss']} (delta={p.get('huber_delta', 0.1)})")
    print(f"  Surge weight: {surge_weight}x")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"],
                                   weight_decay=p.get("weight_decay", 0.01))
    sched_type = p.get("scheduler", "cosine")
    if sched_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=p.get("scheduler_factor", 0.5),
            patience=p.get("scheduler_patience", 5),
            min_lr=1e-6,
        )
        print(f"  Scheduler: ReduceLROnPlateau (factor={p.get('scheduler_factor', 0.5)}, "
              f"patience={p.get('scheduler_patience', 5)})")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p["epochs"])
        print(f"  Scheduler: CosineAnnealingLR")
    grad_clip = p.get("grad_clip", 1.0)

    if p["loss"] == "huber":
        criterion = nn.HuberLoss(delta=p.get("huber_delta", 0.1), reduction='none')
    else:
        criterion = nn.MSELoss(reduction='none')

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, p["epochs"] + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_count = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{p['epochs']} [Train]",
                         leave=False, unit="batch")
        for context, sector, target in train_bar:
            context = context.to(device)
            sector = sector.to(device)
            target = target.to(device)

            pred = model(context, sector)
            per_sample_loss = criterion(pred, target)

            # Apply surge weighting in loss
            weights = torch.where(target >= data_cfg["surge_threshold"],
                                  surge_weight, normal_weight).to(device)
            loss = (per_sample_loss * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item() * context.size(0)
            train_count += context.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= train_count

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_count = 0
        all_preds = []
        all_targets = []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{p['epochs']} [Val]  ",
                       leave=False, unit="batch")
        with torch.no_grad():
            for context, sector, target in val_bar:
                context = context.to(device)
                sector = sector.to(device)
                target = target.to(device)

                pred = model(context, sector)
                per_sample_loss = criterion(pred, target)
                loss = per_sample_loss.mean()

                val_loss += loss.item() * context.size(0)
                val_count += context.size(0)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        val_loss /= val_count
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Evaluate as classifier: "predicted surge" = pred >= threshold
        surge_threshold = data_cfg["surge_threshold"]
        pred_surge = all_preds >= surge_threshold * 0.5  # lower threshold for predicted values
        actual_surge = all_targets >= surge_threshold
        tp = int((pred_surge & actual_surge).sum())
        fp = int((pred_surge & ~actual_surge).sum())
        fn = int((~pred_surge & actual_surge).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        # Correlation
        if len(all_preds) > 1:
            corr = np.corrcoef(all_preds, all_targets)[0, 1]
        else:
            corr = 0.0

        if sched_type == "plateau":
            scheduler.step(val_loss)
            lr_now = optimizer.param_groups[0]['lr']
        else:
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        flag = " *" if val_loss < best_val_loss else f"  (patience {patience_counter+1}/{p['early_stopping_patience']})"
        print(f"Epoch {epoch:02d}/{p['epochs']} | "
              f"Train {train_loss:.4f} | "
              f"Val {val_loss:.4f} corr={corr:.3f} prec={precision:.1%} rec={recall:.1%} | "
              f"LR {lr_now:.2e} | {elapsed:.0f}s{flag}")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_corr": float(corr),
            "val_precision": float(precision), "val_recall": float(recall),
            "val_tp": tp, "val_fp": fp, "val_fn": fn,
            "pred_mean": float(all_preds.mean()), "pred_std": float(all_preds.std()),
            "lr": lr_now, "elapsed": elapsed
        })

        # Early stopping on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": p,
                "model_type": "regressor",
                "epoch": epoch,
                "val_loss": val_loss,
                "val_corr": float(corr),
                "val_precision": float(precision),
                "val_recall": float(recall),
            }, model_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= p["early_stopping_patience"]:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": p,
        "model_type": "regressor",
        "epoch": epoch,
        "val_loss": val_loss,
    }, model_dir / "final_model.pt")

    with open(model_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    total_time = sum(h["elapsed"] for h in history)
    best = min(history, key=lambda h: h["val_loss"])

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time    : {total_time/60:.1f} min")
    print(f"  Epochs run    : {len(history)}/{p['epochs']}")
    print(f"  Best epoch    : {best['epoch']} (val_loss={best['val_loss']:.4f})")
    print(f"  Best corr     : {best['val_corr']:.3f}")
    print(f"  Best precision: {best['val_precision']:.2%}")
    print(f"  Best recall   : {best['val_recall']:.2%}")
    print(f"  Model saved   : {model_dir / 'best_model.pt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
