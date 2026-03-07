"""
V7 PatchTST Inference — Classification model outputs surge probability directly
"""
import numpy as np
import torch
from pathlib import Path

from ..train.train_v7 import PatchTSTClassifier, get_device


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
        self.model_type = checkpoint.get("model_type", "regression")

        self.model = PatchTSTClassifier(
            input_channels=p["input_channels"],
            context_length=p["context_length"],
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
              f"(epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.6f}, "
              f"type={self.model_type})")

    def predict(self, context, current_price=None, surge_threshold=0.15,
                n_samples=30):
        """Predict surge probability.

        For classifier model: MC Dropout on logit, then sigmoid.
        For legacy regression model: falls back to old behavior.

        Args:
            context: numpy array (60, 3) — [close_norm, log_vol_norm, rsi]
            current_price: float (unused for classifier, kept for API compat)
            surge_threshold: float (unused for classifier)
            n_samples: int — MC Dropout samples

        Returns:
            dict with surge_prob, confidence_low/high, etc.
        """
        if self.model is None:
            return {"surge_prob": 0.0, "model_available": False}

        context_t = torch.from_numpy(context).float().unsqueeze(0).to(self.device)

        # MC Dropout: enable dropout during inference
        self.model.train()
        probs = []
        with torch.no_grad():
            for _ in range(n_samples):
                logit = self.model(context_t)  # (1,)
                prob = torch.sigmoid(logit).cpu().item()
                probs.append(prob)

        self.model.eval()
        probs = np.array(probs)

        surge_prob = float(np.mean(probs))
        confidence_low = float(np.percentile(probs, 10))
        confidence_high = float(np.percentile(probs, 90))

        return {
            "surge_prob": surge_prob,
            "confidence_low": confidence_low,
            "confidence_high": confidence_high,
            "model_available": True,
        }
