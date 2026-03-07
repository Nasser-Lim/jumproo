"""
V7 HMM Regime Detector — Rolling Window, Strictly Past

Key v7 changes vs v6:
- fit() uses rolling window, never includes current prediction point
- Once fitted, parameters are frozen until next refit
"""
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Model is not converging")

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False


class RegimeDetector:

    def __init__(self, n_regimes=3, covariance_type="full", n_iter=100,
                 soft_gate_min=0.1, soft_gate_slope=0.2):
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.soft_gate_min = soft_gate_min
        self.soft_gate_slope = soft_gate_slope
        self.model = None
        self.regime_labels = {}
        self.fitted = False
        self.use_fallback = not HAS_HMMLEARN

    def fit(self, returns, volatility, volume_change):
        """Fit HMM on strictly past data within rolling window."""
        returns = np.asarray(returns, dtype=float)
        volatility = np.asarray(volatility, dtype=float)
        volume_change = np.asarray(volume_change, dtype=float)

        mask = np.isfinite(returns) & np.isfinite(volatility) & np.isfinite(volume_change)
        returns = returns[mask]
        volatility = volatility[mask]
        volume_change = volume_change[mask]

        if len(returns) < 30:
            self.use_fallback = True
            self._fit_thresholds(returns, volatility, volume_change)
            return self

        X = np.column_stack([returns, volatility, volume_change])

        if self.use_fallback:
            self._fit_thresholds(returns, volatility, volume_change)
            return self

        try:
            model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=42,
            )
            model.fit(X)
            self.model = model

            means = model.means_
            mean_returns = means[:, 0]
            mean_volatility = means[:, 1]

            surge_idx = int(np.argmax(mean_returns))
            calm_idx = int(np.argmin(mean_volatility))
            volatile_idx = [i for i in range(self.n_regimes)
                            if i not in (surge_idx, calm_idx)]
            volatile_idx = volatile_idx[0] if volatile_idx else calm_idx

            self.regime_labels = {
                surge_idx: "surge_ready",
                calm_idx: "calm",
                volatile_idx: "volatile",
            }
            self.fitted = True
        except Exception:
            self.use_fallback = True
            self._fit_thresholds(returns, volatility, volume_change)

        return self

    def _fit_thresholds(self, returns, volatility, volume_change):
        self.vol_median = float(np.median(volatility)) if len(volatility) > 0 else 0.02
        self.vc_median = float(np.median(volume_change)) if len(volume_change) > 0 else 0.0
        self.fitted = True

    def predict_regime(self, returns, volatility, volume_change):
        returns = np.asarray(returns, dtype=float)
        volatility = np.asarray(volatility, dtype=float)
        volume_change = np.asarray(volume_change, dtype=float)

        if self.use_fallback or self.model is None:
            return self._predict_fallback(returns, volatility, volume_change)

        mask = np.isfinite(returns) & np.isfinite(volatility) & np.isfinite(volume_change)
        r, v, vc = returns[mask], volatility[mask], volume_change[mask]

        if len(r) < 1:
            return {"current_regime": "calm", "is_surge_ready": False,
                    "surge_ready_prob": 0.0, "gate": self.soft_gate_min}

        X = np.column_stack([r, v, vc])
        try:
            probs = self.model.predict_proba(X)
            last_probs = probs[-1]
            current_state = int(np.argmax(last_probs))
            current_regime = self.regime_labels.get(current_state, "volatile")
            is_surge_ready = current_regime == "surge_ready"

            # Find surge_ready state index
            sr_idx = [k for k, v in self.regime_labels.items() if v == "surge_ready"]
            surge_ready_prob = float(last_probs[sr_idx[0]]) if sr_idx else 0.0

            # Soft gate
            if is_surge_ready:
                gate = 1.0
            else:
                gate = self.soft_gate_min + self.soft_gate_slope * surge_ready_prob

            return {
                "current_regime": current_regime,
                "is_surge_ready": is_surge_ready,
                "surge_ready_prob": surge_ready_prob,
                "gate": float(gate),
            }
        except Exception:
            return self._predict_fallback(returns, volatility, volume_change)

    def _predict_fallback(self, returns, volatility, volume_change):
        vol_med = getattr(self, 'vol_median', 0.02)
        vc_med = getattr(self, 'vc_median', 0.0)
        last_vol = float(volatility[-1]) if len(volatility) > 0 else 0
        last_vc = float(volume_change[-1]) if len(volume_change) > 0 else 0

        if last_vol > 2 * vol_med and last_vc > vc_med:
            regime, gate = "surge_ready", 1.0
        elif last_vol > 1.5 * vol_med:
            regime, gate = "volatile", 0.3
        else:
            regime, gate = "calm", self.soft_gate_min

        return {
            "current_regime": regime,
            "is_surge_ready": regime == "surge_ready",
            "surge_ready_prob": 0.33,
            "gate": gate,
        }
