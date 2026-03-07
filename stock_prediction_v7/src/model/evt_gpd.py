"""
V7 EVT/GPD Surge Estimator — Strictly Past, Rolling Window, No Recency Weighting
"""
import numpy as np
from scipy.stats import genpareto


class EVTSurgeEstimator:
    """Extreme Value Theory surge probability using GPD.

    Key v7 changes vs v6:
    - Rolling window (504 days) instead of expanding
    - Uniform weights instead of recency-weighted
    - Strictly uses [t-window : t-1], never includes t
    - No pseudo-sampling / randomness in fitting
    """

    def __init__(self, tail_fraction=0.05, min_exceedances=20,
                 surge_threshold=0.15, rolling_window=504):
        self.tail_fraction = tail_fraction
        self.min_exceedances = min_exceedances
        self.surge_threshold = surge_threshold
        self.rolling_window = rolling_window
        self.shape = None
        self.scale = None
        self.threshold = None
        self.surge_prob = None
        self.method = None
        self.vol_adjustment = 1.0

    def fit(self, returns, volatility=None):
        """Fit GPD on rolling window of returns.

        Args:
            returns: array of daily returns (must already be sliced to [t-window : t-1])
            volatility: array of rolling volatility (same slice)
        """
        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]

        # Enforce rolling window
        if len(returns) > self.rolling_window:
            returns = returns[-self.rolling_window:]
            if volatility is not None:
                volatility = np.asarray(volatility, dtype=float)
                volatility = volatility[np.isfinite(volatility)]
                volatility = volatility[-self.rolling_window:]

        if len(returns) < 30:
            self._set_fallback()
            return self

        # 5-day rolling max
        rm5 = self._rolling_max(returns, 5)
        self._fit_gpd(rm5)

        # 3-day rolling max (secondary)
        rm3 = self._rolling_max(returns, 3)
        surge_prob_3d = self._compute_simple_gpd(rm3)

        # Volatility conditioning
        if volatility is not None:
            self._apply_vol_conditioning(volatility)

        # Combine: 60% primary GPD + 40% secondary GPD
        if self.surge_prob is not None and surge_prob_3d is not None:
            self.surge_prob = 0.6 * self.surge_prob + 0.4 * surge_prob_3d
        elif surge_prob_3d is not None:
            self.surge_prob = surge_prob_3d

        # Apply vol adjustment
        if self.surge_prob is not None:
            self.surge_prob = float(np.clip(self.surge_prob * self.vol_adjustment, 0, 1))

        return self

    def _rolling_max(self, returns, window):
        n = len(returns)
        if n < window:
            return returns.copy()
        return np.array([
            np.max(returns[max(0, i):i + window]) for i in range(n - window + 1)
        ])

    def _fit_gpd(self, rolling_max):
        """Fit GPD with uniform weights."""
        n = len(rolling_max)
        if n < 20:
            self._set_fallback_empirical(rolling_max)
            return

        threshold = np.percentile(rolling_max, 100 * (1 - self.tail_fraction))
        exceedances = rolling_max[rolling_max > threshold] - threshold

        if len(exceedances) < self.min_exceedances:
            self._set_fallback_empirical(rolling_max)
            return

        try:
            shape, loc, scale = genpareto.fit(exceedances, floc=0)
            if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
                self._set_fallback_empirical(rolling_max)
                return

            self.shape = shape
            self.scale = scale
            self.threshold = threshold
            self.method = "gpd"

            if self.surge_threshold > threshold:
                excess = self.surge_threshold - threshold
                prob = self.tail_fraction * genpareto.sf(excess, shape, scale=scale)
                self.surge_prob = float(np.clip(prob, 0, 1))
            else:
                self.surge_prob = float(np.mean(rolling_max >= self.surge_threshold))

        except Exception:
            self._set_fallback_empirical(rolling_max)

    def _compute_simple_gpd(self, rolling_max):
        """Fit simple GPD on secondary rolling max. Returns prob or None."""
        n = len(rolling_max)
        if n < 20:
            return float(np.mean(rolling_max >= self.surge_threshold)) if n > 0 else 0.0

        threshold = np.percentile(rolling_max, 100 * (1 - self.tail_fraction))
        exceedances = rolling_max[rolling_max > threshold] - threshold

        if len(exceedances) < self.min_exceedances:
            return float(np.mean(rolling_max >= self.surge_threshold))

        try:
            shape, loc, scale = genpareto.fit(exceedances, floc=0)
            if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
                return float(np.mean(rolling_max >= self.surge_threshold))

            if self.surge_threshold > threshold:
                excess = self.surge_threshold - threshold
                prob = self.tail_fraction * genpareto.sf(excess, shape, scale=scale)
                return float(np.clip(prob, 0, 1))
            else:
                return float(np.mean(rolling_max >= self.surge_threshold))
        except Exception:
            return float(np.mean(rolling_max >= self.surge_threshold))

    def _apply_vol_conditioning(self, volatility):
        volatility = np.asarray(volatility, dtype=float)
        volatility = volatility[np.isfinite(volatility)]
        if len(volatility) < 20:
            self.vol_adjustment = 1.0
            return

        current_vol = volatility[-1]
        median_vol = np.median(volatility)
        if median_vol <= 0:
            self.vol_adjustment = 1.0
            return

        vol_ratio = current_vol / median_vol
        if vol_ratio > 2.0:
            self.vol_adjustment = min(1.5, 0.8 + 0.35 * vol_ratio)
        elif vol_ratio > 1.0:
            self.vol_adjustment = 0.8 + 0.2 * vol_ratio
        else:
            self.vol_adjustment = max(0.5, 0.8 * vol_ratio)

    def _set_fallback(self):
        self.method = "fallback"
        self.surge_prob = 0.0
        self.shape = 0.0
        self.scale = 0.01

    def _set_fallback_empirical(self, rolling_max):
        self.method = "empirical"
        self.surge_prob = float(np.mean(rolling_max >= self.surge_threshold))
        self.shape = 0.0
        self.scale = 0.01

    def predict_surge_probability(self):
        return {
            "surge_prob": self.surge_prob if self.surge_prob is not None else 0.0,
            "vol_adjustment": self.vol_adjustment,
            "gpd_shape": self.shape if self.shape is not None else 0.0,
            "gpd_scale": self.scale if self.scale is not None else 0.01,
            "threshold": self.threshold,
            "method": self.method or "not_fitted",
        }
