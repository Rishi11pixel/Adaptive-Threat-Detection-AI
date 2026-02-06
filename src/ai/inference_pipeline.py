import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore

class AnomalyInferencePipeline:
    def __init__(self):
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
        self.is_fitted = False

    def fit(self, X):
        self.iso.fit(X)
        self.lof.fit(X)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before prediction")

        iso_pred = np.where(self.iso.predict(X) == -1, 1, 0)

        z_scores = np.abs(zscore(X))
        z_pred = (z_scores.max(axis=1) > 3).astype(int)

        lof_pred = np.where(self.lof.predict(X) == -1, 1, 0)

        votes = iso_pred + z_pred + lof_pred
        fused_pred = (votes >= 2).astype(int)

        return fused_pred, votes
