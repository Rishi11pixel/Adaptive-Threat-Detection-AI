import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore

# Load data
df = pd.read_csv("data/processed_sensor_data.csv")
X = df.drop("label", axis=1)
y_true = df["label"]

# -------- Model 1: Isolation Forest --------
iso = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
iso.fit(X)
iso_pred = np.where(iso.predict(X) == -1, 1, 0)

# -------- Model 2: Z-score detector --------
z_scores = np.abs(zscore(X))
z_pred = (z_scores.max(axis=1) > 3).astype(int)

# -------- Model 3: Local Outlier Factor --------
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_pred = np.where(lof.fit_predict(X) == -1, 1, 0)

# -------- Fusion Logic (Voting) --------
votes = iso_pred + z_pred + lof_pred
fused_pred = (votes >= 2).astype(int)

# -------- Analysis --------
print("Model disagreement statistics:")
print("Isolation Forest anomalies:", iso_pred.sum())
print("Z-score anomalies:", z_pred.sum())
print("LOF anomalies:", lof_pred.sum())
print("Fused anomalies:", fused_pred.sum())

disagreement = np.sum(
    (iso_pred != z_pred) |
    (iso_pred != lof_pred) |
    (z_pred != lof_pred)
)

print("Disagreement count:", disagreement)
