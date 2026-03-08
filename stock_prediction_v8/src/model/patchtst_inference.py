# -*- coding: utf-8 -*-
"""
V8 PatchTST Inference — Regression model outputs predicted max 5-day return.
MC Dropout for uncertainty estimation.
"""
import numpy as np
import torch
from pathlib import Path

from ..train.train_v8 import PatchTSTRegressor, get_device


class PatchTSTPredictor:

    def __init__(self, model_path=None, device="auto"):
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models" / "patchtst" / "best_model.pt"

        self.device = get_device(device)
        self.model = None
        self.model_path = Path(model_path)

        if self.model_path.exists():
            self._load_model()
        else:
            print(f"WARNING: Model not found at {model_path}. "
                  f"Run train_v8.py first.")

    def _load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device,
                                weights_only=False)
        p = checkpoint["config"]
        self.model_type = checkpoint.get("model_type", "regressor")

        self.model = PatchTSTRegressor(
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
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"PatchTST V8 loaded from {self.model_path} "
              f"(epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.6f}, "
              f"type={self.model_type})")

    def predict(self, context, sector_id=0, n_samples=30, surge_threshold=0.15):
        """Predict max 5-day return with MC Dropout uncertainty.

        Args:
            context: numpy array (60, 8) — 8 channel time-series context
            sector_id: int — sector index for embedding
            n_samples: int — MC Dropout samples
            surge_threshold: float — threshold for surge classification

        Returns:
            dict with predicted_return, surge_prob, confidence_low/high
        """
        if self.model is None:
            return {"predicted_return": 0.0, "surge_prob": 0.0, "model_available": False}

        context_t = torch.from_numpy(context).float().unsqueeze(0).to(self.device)
        sector_t = torch.tensor([sector_id], dtype=torch.long).to(self.device)

        # MC Dropout: enable dropout during inference
        self.model.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(context_t, sector_t)
                preds.append(pred.cpu().item())

        self.model.eval()
        preds = np.array(preds)

        predicted_return = float(np.mean(preds))
        confidence_low = float(np.percentile(preds, 10))
        confidence_high = float(np.percentile(preds, 90))

        # Surge probability: fraction of MC samples predicting above threshold
        surge_prob = float((preds >= surge_threshold * 0.5).mean())

        return {
            "predicted_return": predicted_return,
            "surge_prob": surge_prob,
            "confidence_low": confidence_low,
            "confidence_high": confidence_high,
            "model_available": True,
        }
