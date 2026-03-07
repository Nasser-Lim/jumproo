"""
V7 PatchTST Inference — Load trained model and compute surge probability
"""
import numpy as np
import torch
from pathlib import Path

from ..train.train_v7 import PatchTSTModel, get_device


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
                  f"Run train_v7.py first or git pull from GPU machine.")

    def _load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device,
                                weights_only=False)
        p = checkpoint["config"]
        self.model = PatchTSTModel(
            input_channels=p["input_channels"],
            context_length=p["context_length"],
            prediction_length=p["prediction_length"],
            patch_length=p["patch_length"],
            stride=p["stride"],
            d_model=p["d_model"],
            n_heads=p["n_heads"],
            n_layers=p["n_layers"],
            dropout=p["dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"PatchTST loaded from {self.model_path} "
              f"(epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.6f})")

    def predict(self, context, current_price, surge_threshold=0.15,
                n_samples=30):
        """Predict surge probability using MC Dropout.

        Args:
            context: numpy array (60, 3) — [close_norm, log_vol_norm, rsi]
            current_price: float — latest close price (for denormalization)
            surge_threshold: float — 15% threshold
            n_samples: int — MC Dropout samples

        Returns:
            dict with surge_prob, forecast_return, etc.
        """
        if self.model is None:
            return {"surge_prob": 0.0, "forecast_return": 0.0,
                    "model_available": False}

        context_t = torch.from_numpy(context).float().unsqueeze(0).to(self.device)

        # MC Dropout: enable dropout during inference
        self.model.train()  # enable dropout
        forecasts = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(context_t)  # (1, 5)
                forecasts.append(pred.cpu().numpy()[0])

        self.model.eval()
        forecasts = np.array(forecasts)  # (n_samples, 5)

        # Forecasts are normalized close prices (relative to current_price)
        # Max return within 5-day window for each sample
        max_returns = np.max(forecasts, axis=1) - 1.0  # subtract 1 since normalized

        surge_prob = float(np.mean(max_returns >= surge_threshold))
        mean_max_return = float(np.mean(max_returns))
        median_forecast = np.median(forecasts, axis=0)

        return {
            "surge_prob": surge_prob,
            "forecast_return": mean_max_return,
            "forecast_5d": (median_forecast * current_price).tolist(),
            "confidence_low": float(np.percentile(max_returns, 10)),
            "confidence_high": float(np.percentile(max_returns, 90)),
            "model_available": True,
        }
