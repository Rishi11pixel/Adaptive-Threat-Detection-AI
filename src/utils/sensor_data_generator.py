import numpy as np
import pandas as pd

def generate_sensor_data(samples=1000, anomaly_ratio=0.05):
    """
    Generates synthetic multi-sensor data with realistic near-boundary anomalies.
    Anomalies partially overlap with normal behavior to avoid trivial detection.
    """

    np.random.seed(42)

    # -----------------------------
    # Normal operational data
    # -----------------------------
    motion = np.random.normal(loc=0.5, scale=0.1, size=samples)
    proximity = np.random.normal(loc=1.0, scale=0.2, size=samples)
    noise = np.random.normal(loc=30, scale=5, size=samples)

    labels = np.zeros(samples)

    # -----------------------------
    # Inject anomalies (near-boundary)
    # -----------------------------
    anomaly_count = int(samples * anomaly_ratio)
    anomaly_indices = np.random.choice(samples, anomaly_count, replace=False)

    # Type 1: Subtle motion + noise increase (evasive movement)
    motion[anomaly_indices] += np.random.normal(loc=0.25, scale=0.08, size=anomaly_count)
    noise[anomaly_indices] += np.random.normal(loc=8, scale=3, size=anomaly_count)

    # Type 2: Sensor confusion (proximity slightly off)
    proximity[anomaly_indices] += np.random.normal(loc=0.15, scale=0.05, size=anomaly_count)

    labels[anomaly_indices] = 1

    # -----------------------------
    # Create DataFrame
    # -----------------------------
    df = pd.DataFrame({
        "motion": motion,
        "proximity": proximity,
        "noise": noise,
        "label": labels
    })

    return df


if __name__ == "__main__":
    df = generate_sensor_data(samples=1000, anomaly_ratio=0.05)
    df.to_csv("data/sensor_data.csv", index=False)
    print("Realistic sensor data with near-boundary anomalies generated.")
