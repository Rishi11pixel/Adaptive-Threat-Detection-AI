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
pipeline = AnomalyInferencePipeline()
pipeline.fit(X_train)

print("Starting sensor stream...\n")

# -------------------------------
# Simulated real-time stream
# -------------------------------
BATCH_SIZE = 5

for i in range(0, len(X_stream), BATCH_SIZE):
    batch = X_stream.iloc[i:i + BATCH_SIZE]

    # Safety check (important in real streams)
    if batch.empty:
        continue

    preds, votes = pipeline.predict(batch)

    anomaly_count = int(preds.sum())

    if anomaly_count > 0:
        print(
            f"[ALERT] Batch {i // BATCH_SIZE} | "
            f"Anomalies: {anomaly_count} | "
            f"Votes: {votes.tolist()}"
        )
    else:
        print(f"[INFO] Batch {i // BATCH_SIZE} | Normal")

    time.sleep(0.5)  # simulate sensor delay
