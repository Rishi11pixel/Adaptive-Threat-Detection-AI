import pandas as pd
import time
from src.ai.inference_pipeline import AnomalyInferencePipeline

# -------------------------------
# Load processed sensor data
# -------------------------------
df = pd.read_csv("data/processed_sensor_data.csv")

# Separate features and labels
X = df.drop(columns=["label"])
y_true = df["label"]

# -------------------------------
# Train / Stream split
# -------------------------------
train_size = int(0.6 * len(X))

X_train = X.iloc[:train_size].reset_index(drop=True)
X_stream = X.iloc[train_size:].reset_index(drop=True)
y_stream = y_true.iloc[train_size:].reset_index(drop=True)

# -------------------------------
# Initialize and train pipeline
# -------------------------------
pipeline = AnomalyInferencePipeline(
    window_size=3,        # short-term memory
    alert_threshold=2     # require persistence
)
pipeline.fit(X_train)

print("Starting sensor stream with temporal smoothing...\n")

# -------------------------------
# Simulated real-time stream
# -------------------------------
BATCH_SIZE = 5

for i in range(0, len(X_stream), BATCH_SIZE):
    batch = X_stream.iloc[i:i + BATCH_SIZE]

    # Safety check (important in real streams)
    if batch.empty:
        continue

    fused_preds, smoothed_alert = pipeline.predict(batch)

    anomaly_count = int(fused_preds.sum())

    # Raw anomaly info (for debugging / logs)
    if anomaly_count > 0:
        print(
            f"[RAW] Batch {i // BATCH_SIZE} | "
            f"Detected anomalies: {anomaly_count}"
        )
    else:
        print(f"[RAW] Batch {i // BATCH_SIZE} | Normal")

    # Operational alert (what robot / operator sees)
    if smoothed_alert:
        print(
            f"[SMOOTHED ALERT] Batch {i // BATCH_SIZE} | "
            f"Sustained anomalous behavior detected"
        )

    time.sleep(0.5)  # simulate sensor delay
