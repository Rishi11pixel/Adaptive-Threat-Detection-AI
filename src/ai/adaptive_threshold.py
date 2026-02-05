import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/processed_sensor_data.csv")
X = df.drop("label", axis=1)
y_true = df["label"]

# Train without assuming contamination
model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42
)
model.fit(X)

# Get anomaly scores (lower = more anomalous)
scores = model.decision_function(X)

# Adaptive threshold using percentile
threshold = np.percentile(scores, 10)  # intentionally arbitrary

y_pred = (scores < threshold).astype(int)

# Count instability
print("Adaptive Threshold:", threshold)
print("Predicted anomalies:", y_pred.sum())
print("True anomalies:", y_true.sum())
plt.hist(scores, bins=50)
plt.axvline(threshold, color='red', linestyle='--', label='Adaptive Threshold')
plt.title("Anomaly Score Distribution (Unstable by Design)")
plt.legend()
plt.show()
