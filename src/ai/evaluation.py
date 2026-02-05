import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# -----------------------------
# Load processed data
# -----------------------------
df = pd.read_csv("data/processed_sensor_data.csv")

X = df.drop("label", axis=1)
y = df["label"]

# -----------------------------
# Train / Test Split
# Prevents data leakage
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train Isolation Forest
# -----------------------------
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

model.fit(X_train)

# -----------------------------
# Predict on TEST data only
# -----------------------------
y_pred = model.predict(X_test)

# Convert IF output to binary labels
# -1 → anomaly (1), 1 → normal (0)
y_pred = np.where(y_pred == -1, 1, 0)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")
plt.title("Isolation Forest – Test Set Confusion Matrix")
plt.tight_layout()
plt.savefig("dashboards/confusion_matrix.png")
plt.show()

# -----------------------------
# Print results
# -----------------------------
print("Confusion Matrix (Test Set):")
print(cm)

tn, fp, fn, tp = cm.ravel()
print("\nInterpretation:")
print(f"True Negatives  (Normal detected correctly): {tn}")
print(f"False Positives (False alarms):             {fp}")
print(f"False Negatives (Missed threats):           {fn}")
print(f"True Positives  (Threats detected):         {tp}")
