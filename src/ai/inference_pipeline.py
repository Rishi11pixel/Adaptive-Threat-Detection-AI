import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore

class AnomalyInferencePipeline:
    def __init__(self, window_size=3, alert_threshold=2):
        self.iso = IsolationForest(
            n_estimators=100,
            contamination="auto",
            random_state=42
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.05,
            novelty=True
        )

        # ---- NEW: temporal smoothing parameters ----
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.recent_alerts = deque(maxlen=window_size)

        self.is_fitted = False

    def fit(self, X):
        self.iso.fit(X)
        self.lof.fit(X)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before prediction")

        # ---- Individual detectors ----
        iso_pred = np.where(self.iso.predict(X) == -1, 1, 0)

        z_scores = np.abs(zscore(X))
        z_pred = (z_scores.max(axis=1) > 3).astype(int)

        lof_pred = np.where(self.lof.predict(X) == -1, 1, 0)

        # ---- Fusion ----
        votes = iso_pred + z_pred + lof_pred
        fused_pred = (votes >= 2).astype(int)

        # ---- NEW: batch-level alert memory ----
        batch_alert = int(fused_pred.sum() > 0)
        self.recent_alerts.append(batch_alert)

        smoothed_alert = sum(self.recent_alerts) >= self.alert_threshold

        return fused_pred, smoothed_alert
