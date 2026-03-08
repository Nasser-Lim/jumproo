# -*- coding: utf-8 -*-
"""
V8 Backtest — PatchTST Regressor on Val/Test set.

Evaluates:
  1. PatchTST-only: various thresholds on predicted_return
  2. Stats-only (placeholder — uses val set surge_label as proxy)
  3. Combined scoring (stats 70% + patchtst 30%) — needs stat pipeline
  
Usage:
  python stock_prediction_v8/scripts/backtest_v8.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train.train_v8 import PatchTSTRegressor, SurgeRegressionDataset, get_device


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    p = checkpoint["config"]
    model = PatchTSTRegressor(
        ts_channels=p.get("ts_channels", 8),
        context_length=p["context_length"],
        patch_length=p["patch_length"],
        stride=p["stride"],
        d_model=p["d_model"],
        n_heads=p["n_heads"],
        n_layers=p["n_layers"],
        dropout=p["dropout"],
        num_sectors=p.get("num_sectors", 170),
        sector_embed_dim=p.get("sector_embed_dim", 16),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded: epoch={checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f}")
    return model, p


def run_inference(model, npz_path, device, batch_size=512):
    """Run model on all samples in npz, return preds and targets."""
    dataset = SurgeRegressionDataset(npz_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []
    all_sectors = []

    with torch.no_grad():
        for context, sector, target in loader:
            context = context.to(device)
            sector = sector.to(device)
            pred = model(context, sector)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())
            all_sectors.extend(sector.cpu().numpy())

    return np.array(all_preds), np.array(all_targets), np.array(all_sectors)


def evaluate_thresholds(preds, targets, surge_threshold=0.15, split_name="Val"):
    """Evaluate precision/recall at various predicted return thresholds."""
    actual_surge = targets >= surge_threshold
    base_rate = actual_surge.mean()
    total_surge = actual_surge.sum()
    n = len(targets)

    print(f"\n{'='*65}")
    print(f"  {split_name} Set Evaluation  (N={n:,}, surge rate={base_rate:.2%}, surge count={total_surge})")
    print(f"{'='*65}")
    print(f"  Pred correlation with target: {np.corrcoef(preds, targets)[0,1]:.4f}")
    print(f"  Pred stats: mean={preds.mean():.4f}, std={preds.std():.4f}, "
          f"min={preds.min():.4f}, max={preds.max():.4f}")
    print(f"  Target stats: mean={targets.mean():.4f}, std={targets.std():.4f}")
    print()
    print(f"  {'Threshold':>10} | {'Signals':>8} | {'Sig%':>6} | {'Precision':>10} | {'Recall':>8} | {'AvgRet(sig)':>12} | {'AvgRet(actual)':>15}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*12}-+-{'-'*15}")

    thresholds = [0.03, 0.05, 0.07, 0.10, 0.075, 0.125]
    thresholds = sorted(set(thresholds))

    results = []
    for thr in thresholds:
        pred_surge = preds >= thr
        signals = pred_surge.sum()
        sig_pct = signals / n
        tp = int((pred_surge & actual_surge).sum())
        fp = int((pred_surge & ~actual_surge).sum())
        fn = int((~pred_surge & actual_surge).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        avg_ret_sig = targets[pred_surge].mean() if signals > 0 else 0.0
        avg_ret_actual = targets[actual_surge].mean() if total_surge > 0 else 0.0

        print(f"  {thr:>10.3f} | {signals:>8,} | {sig_pct:>6.2%} | {precision:>10.2%} | {recall:>8.2%} | {avg_ret_sig:>12.4f} | {avg_ret_actual:>15.4f}")
        results.append(dict(threshold=thr, signals=int(signals), precision=precision,
                            recall=recall, avg_ret_sig=float(avg_ret_sig)))

    print(f"\n  Base rate: {base_rate:.2%} | Total actual surges: {total_surge}")
    return results


def main():
    root = Path(__file__).parent.parent
    data_dir = root / "data" / "processed"
    model_path = root / "models" / "patchtst" / "best_model.pt"

    device = get_device("auto")
    print(f"\n  Device: {device}")

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    print(f"\n  Loading model from {model_path}...")
    model, config = load_model(model_path, device)

    # --- Val set ---
    val_path = data_dir / "val.npz"
    if val_path.exists():
        print(f"\n  Running inference on val set...")
        val_preds, val_targets, val_sectors = run_inference(model, val_path, device)
        val_results = evaluate_thresholds(val_preds, val_targets, split_name="Val")
        np.savez(data_dir / "val_preds.npz", preds=val_preds, targets=val_targets, sectors=val_sectors)
        print(f"\n  Val predictions saved to {data_dir / 'val_preds.npz'}")
    else:
        print(f"  WARNING: val.npz not found at {val_path}")

    # --- Test set ---
    test_path = data_dir / "test.npz"
    if test_path.exists():
        print(f"\n  Running inference on test set...")
        test_preds, test_targets, test_sectors = run_inference(model, test_path, device)
        test_results = evaluate_thresholds(test_preds, test_targets, split_name="Test")
        np.savez(data_dir / "test_preds.npz", preds=test_preds, targets=test_targets, sectors=test_sectors)
        print(f"\n  Test predictions saved to {data_dir / 'test_preds.npz'}")
    else:
        print(f"  WARNING: test.npz not found at {test_path}")

    # Save summary
    summary = {
        "model_path": str(model_path),
        "val_results": val_results if val_path.exists() else [],
        "test_results": test_results if test_path.exists() else [],
    }
    out_path = root / "data" / "processed" / "backtest_v8_patchtst.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {out_path}")


if __name__ == "__main__":
    main()
