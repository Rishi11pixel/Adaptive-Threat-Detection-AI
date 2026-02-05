import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -------------------------------
# Load processed sensor data
# -------------------------------
df = pd.read_csv("data/processed_sensor_data.csv")

# Features & labels
X = df.drop("label", axis=1)
y = df["label"]

# -------------------------------
# Train–Test Split (NO LEAKAGE)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -------------------------------
# Train ONLY on normal data
# (Real-world anomaly detection)
# -------------------------------
X_train_normal = X_train[y_train == 0]

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X_train_normal)

# -------------------------------
# Predict on unseen test data
# -------------------------------
y_pred = model.predict(X_test)

# Isolation Forest outputs:
#  1  -> normal
# -1 -> anomaly
y_pred = np.where(y_pred == -1, 1, 0)

# -------------------------------
# Evaluation
# -------------------------------
print("\n=== Isolation Forest Anomaly Detection Results ===\n")
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
