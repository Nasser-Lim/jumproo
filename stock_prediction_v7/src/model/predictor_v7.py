"""
V7 Predictor — PatchTST + Statistical Filter (EVT + Hawkes + HMM + Volume)

Architecture:
  PatchTST forecast_prob ──┐
                           ├──→ final_score
  Stat filter score ───────┘

Stat filter = EVT * Hawkes * HMM_gate * volume_filter
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from .evt_gpd import EVTSurgeEstimator
from .hawkes_timing import HawkesTimingCorrector
from .hmm_regime import RegimeDetector


class SurgePredictor:

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "v7_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        stats = self.cfg["stats"]
        self.rolling_window = stats["rolling_window"]
        self.refit_interval = stats["refit_interval"]

        evt_cfg = stats["evt"]
        self.evt = EVTSurgeEstimator(
            tail_fraction=evt_cfg["tail_fraction"],
            min_exceedances=evt_cfg["min_exceedances"],
            surge_threshold=evt_cfg["surge_definition"],
            rolling_window=self.rolling_window,
        )

        hawkes_cfg = stats["hawkes"]
        self.hawkes = HawkesTimingCorrector(
            decay_beta=hawkes_cfg["decay_beta"],
            surge_lookback_days=hawkes_cfg["cluster_lookback"],
            surge_definition=hawkes_cfg["surge_definition"],
            volume_surge_definition=hawkes_cfg["volume_surge_definition"],
            rolling_window=self.rolling_window,
        )

        hmm_cfg = stats["hmm"]
        self.hmm = RegimeDetector(
            n_regimes=hmm_cfg["n_states"],
            soft_gate_min=hmm_cfg["soft_gate_min"],
            soft_gate_slope=hmm_cfg["soft_gate_slope"],
        )

        vol_cfg = stats["volume_filter"]
        self.vol_short = vol_cfg["short_window"]
        self.vol_long = vol_cfg["long_window"]
        self.vol_threshold = vol_cfg["threshold"]
        self.vol_penalty = vol_cfg["penalty"]

        scoring = self.cfg["scoring"]
        self.patchtst_weight = scoring["patchtst_weight"]
        self.stat_weight = scoring["stat_weight"]
        self.evt_in_stat = scoring["evt_in_stat"]
        self.hawkes_in_stat = scoring["hawkes_in_stat"]

        signals = self.cfg["signals"]
        self.th_strong_buy = signals["strong_buy"]
        self.th_buy = signals["buy"]
        self.th_watch = signals["watch"]

        self._last_refit = -self.refit_interval  # force initial fit

    def fit_stats(self, returns, volatility, volume_change, current_idx):
        """Fit statistical models if refit is due. Uses strictly past data."""
        if current_idx - self._last_refit < self.refit_interval:
            return

        # Slice to [current_idx - rolling_window : current_idx] (excludes current_idx)
        start = max(0, current_idx - self.rolling_window)
        ret_slice = returns[start:current_idx]
        vol_slice = volatility[start:current_idx]
        vc_slice = volume_change[start:current_idx]

        self.evt.fit(ret_slice, volatility=vol_slice)
        self.hawkes.fit(ret_slice, volume_change=vc_slice)
        self.hmm.fit(ret_slice, vol_slice, vc_slice)
        self._last_refit = current_idx

    def predict_stat(self, returns, volatility, volume_change, volume_raw,
                     current_idx):
        """Compute statistical score for a single prediction point.

        All data arrays are full-length; slicing to strictly past is done here.
        """
        start = max(0, current_idx - self.rolling_window)
        ret_slice = returns[start:current_idx]
        vol_slice = volatility[start:current_idx]
        vc_slice = volume_change[start:current_idx]

        # EVT
        evt_result = self.evt.predict_surge_probability()
        evt_prob = evt_result["surge_prob"]

        # Hawkes
        hawkes_result = self.hawkes.compute_intensity(ret_slice, volume_change=vc_slice)
        hawkes_score = hawkes_result["hawkes_score"]

        # HMM gate
        hmm_result = self.hmm.predict_regime(ret_slice, vol_slice, vc_slice)
        gate = hmm_result["gate"]

        # Volume filter
        vol_filter = self._volume_filter(volume_raw, current_idx)

        # Stat score
        raw_stat = (self.evt_in_stat * evt_prob + self.hawkes_in_stat * hawkes_score)
        raw_stat *= gate
        raw_stat *= vol_filter
        stat_score = float(np.clip(raw_stat, 0, 1))

        return {
            "stat_score": stat_score,
            "evt_prob": evt_prob,
            "hawkes_score": hawkes_score,
            "gate": gate,
            "vol_filter": vol_filter,
            "regime": hmm_result["current_regime"],
            "cluster_density": hawkes_result["cluster_density"],
            "vp_convergence": hawkes_result["vp_convergence"],
        }

    def combine_scores(self, patchtst_prob, stat_result):
        """Combine PatchTST probability with statistical score."""
        stat_score = stat_result["stat_score"]
        final = self.patchtst_weight * patchtst_prob + self.stat_weight * stat_score
        final = float(np.clip(final, 0, 1))

        if final >= self.th_strong_buy:
            signal = "STRONG_BUY"
        elif final >= self.th_buy:
            signal = "BUY"
        elif final >= self.th_watch:
            signal = "WATCH"
        else:
            signal = "NEUTRAL"

        return {
            "final_score": final,
            "signal": signal,
            "patchtst_prob": patchtst_prob,
            **stat_result,
        }

    def predict_stat_only(self, returns, volatility, volume_change, volume_raw,
                          current_idx):
        """Stat-only prediction (when PatchTST is not available)."""
        stat_result = self.predict_stat(
            returns, volatility, volume_change, volume_raw, current_idx
        )
        score = stat_result["stat_score"]

        if score >= self.th_strong_buy:
            signal = "STRONG_BUY"
        elif score >= self.th_buy:
            signal = "BUY"
        elif score >= self.th_watch:
            signal = "WATCH"
        else:
            signal = "NEUTRAL"

        return {
            "final_score": score,
            "signal": signal,
            "patchtst_prob": None,
            **stat_result,
        }

    def _volume_filter(self, volume_raw, current_idx):
        """Check if recent volume is elevated. Returns 1.0 (pass) or penalty."""
        if volume_raw is None or current_idx < self.vol_long:
            return 1.0

        short_avg = np.mean(volume_raw[max(0, current_idx - self.vol_short):current_idx])
        long_avg = np.mean(volume_raw[max(0, current_idx - self.vol_long):current_idx])

        if long_avg <= 0:
            return 1.0

        ratio = short_avg / long_avg
        if ratio >= self.vol_threshold:
            return 1.0
        else:
            return self.vol_penalty
