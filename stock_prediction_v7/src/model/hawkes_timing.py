"""
V7 Hawkes Process — Strictly Past Data, Rolling Window

Key v7 changes vs v6:
- fit() and compute_intensity() use [t-window : t-1], never t
- cluster_density counted from [t-lookback : t-1]
- Rolling window (504 days) instead of expanding
"""
import numpy as np
from scipy.optimize import minimize


class HawkesTimingCorrector:

    def __init__(self, decay_beta=0.1, surge_lookback_days=60,
                 surge_definition=0.10, volume_surge_definition=2.0,
                 rolling_window=504):
        self.decay_beta = decay_beta
        self.surge_lookback_days = surge_lookback_days
        self.surge_definition = surge_definition
        self.volume_surge_definition = volume_surge_definition
        self.rolling_window = rolling_window
        # Hawkes params (defaults, updated by fit)
        self.mu_p = 0.01
        self.alpha_p = 0.05
        self.beta_p = decay_beta
        self.mu_v = 0.02
        self.alpha_v = 0.05
        self.beta_v = decay_beta
        self.alpha_vp = 0.03
        self.beta_vp = decay_beta * 0.5

    def fit(self, returns, volume_change=None):
        """Fit Hawkes parameters on strictly past data.

        Args:
            returns: daily returns array (already sliced to [t-window : t-1])
            volume_change: volume change ratios (same slice)
        """
        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]

        if len(returns) > self.rolling_window:
            returns = returns[-self.rolling_window:]

        T = len(returns)
        price_events = np.where(returns >= self.surge_definition)[0].astype(float)
        self._fit_univariate(price_events, T, 'price')

        if volume_change is not None:
            volume_change = np.asarray(volume_change, dtype=float)
            volume_change = volume_change[np.isfinite(volume_change)]
            if len(volume_change) > self.rolling_window:
                volume_change = volume_change[-self.rolling_window:]
            vol_events = np.where(volume_change >= self.volume_surge_definition)[0].astype(float)
            self._fit_univariate(vol_events, T, 'volume')

            if len(price_events) >= 5 and len(vol_events) >= 5:
                self._fit_cross_excitation(price_events, vol_events)

        return self

    def _fit_univariate(self, event_times, T, event_type):
        if len(event_times) < 5:
            return

        def neg_log_likelihood(params):
            mu, alpha, beta = params
            if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
                return 1e10
            n = len(event_times)
            ll = 0.0
            for i in range(n):
                intensity = mu
                for j in range(max(0, i - 50), i):
                    dt = event_times[i] - event_times[j]
                    intensity += alpha * np.exp(-beta * dt)
                if intensity <= 0:
                    return 1e10
                ll += np.log(intensity)
            integral = mu * T
            for j in range(n):
                integral += (alpha / beta) * (1 - np.exp(-beta * (T - event_times[j])))
            return -(ll - integral)

        n_events = len(event_times)
        mu0 = n_events / T
        try:
            result = minimize(
                neg_log_likelihood,
                x0=[mu0, 0.05, self.decay_beta],
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if result.success and result.fun < 1e9:
                mu, alpha, beta = result.x
                if mu > 0 and alpha >= 0 and beta > 0 and alpha < beta:
                    if event_type == 'price':
                        self.mu_p, self.alpha_p, self.beta_p = mu, alpha, beta
                    else:
                        self.mu_v, self.alpha_v, self.beta_v = mu, alpha, beta
        except Exception:
            pass

    def _fit_cross_excitation(self, price_events, vol_events):
        follow_count = 0
        window = 5
        for ve in vol_events:
            for pe in price_events:
                dt = pe - ve
                if 0 < dt <= window:
                    follow_count += 1
                    break
        follow_rate = follow_count / max(len(vol_events), 1)
        self.alpha_vp = float(np.clip(follow_rate * 0.15, 0.01, 0.2))
        self.beta_vp = self.decay_beta * 0.5

    def compute_intensity(self, returns, volume_change=None):
        """Compute Hawkes intensity and features from strictly past data.

        Args:
            returns: daily returns (already sliced to [t-window : t-1])
            volume_change: volume change ratios (same slice)
        """
        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]
        if len(returns) > self.rolling_window:
            returns = returns[-self.rolling_window:]
        T = len(returns)

        price_events = np.where(returns >= self.surge_definition)[0]
        vol_events = np.array([], dtype=int)
        if volume_change is not None:
            volume_change = np.asarray(volume_change, dtype=float)
            vc = volume_change[np.isfinite(volume_change)]
            if len(vc) > self.rolling_window:
                vc = vc[-self.rolling_window:]
            vol_events = np.where(vc >= self.volume_surge_definition)[0]

        # Price intensity
        price_intensity = self.mu_p
        for idx in price_events:
            dt = T - idx
            if dt > 0:
                price_intensity += self.alpha_p * np.exp(-self.beta_p * dt)
        for idx in vol_events:
            dt = T - idx
            if dt > 0:
                price_intensity += self.alpha_vp * np.exp(-self.beta_vp * dt)

        price_ratio = price_intensity / self.mu_p if self.mu_p > 0 else 1.0

        # Cluster features (strictly past)
        lookback = min(self.surge_lookback_days, T)
        recent_price = price_events[price_events >= (T - lookback)]
        recent_vol = vol_events[vol_events >= (T - lookback)]
        n_recent_price = len(recent_price)

        if lookback > 0:
            cluster_density = n_recent_price / (lookback / 20.0)
        else:
            cluster_density = 0.0

        # Days since last surge
        if len(price_events) > 0:
            days_since_last = int(T - price_events[-1])
        else:
            days_since_last = T

        # VP convergence
        vp_convergence = 0.0
        if n_recent_price > 0 and len(recent_vol) > 0:
            overlap = 0
            for pe in recent_price:
                for ve in recent_vol:
                    if abs(int(pe) - int(ve)) <= 3:
                        overlap += 1
                        break
            vp_convergence = overlap / max(n_recent_price, 1)

        # Composite score
        density_score = min(cluster_density / 2.0, 1.0)
        vp_score = float(np.clip(vp_convergence, 0, 1))
        intensity_score = min(price_ratio / 3.0, 1.0)

        if days_since_last <= 5:
            recency_score = 1.0
        elif days_since_last <= 15:
            recency_score = 0.6
        elif days_since_last <= 30:
            recency_score = 0.3
        else:
            recency_score = 0.0

        hawkes_score = (0.40 * density_score + 0.25 * vp_score +
                        0.20 * recency_score + 0.15 * intensity_score)

        return {
            "hawkes_score": float(hawkes_score),
            "cluster_density": float(cluster_density),
            "vp_convergence": float(vp_convergence),
            "intensity_ratio": float(price_ratio),
            "days_since_last_surge": days_since_last,
            "n_recent_surges": n_recent_price,
        }
