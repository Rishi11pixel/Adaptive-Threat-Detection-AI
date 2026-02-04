import numpy as np
import pandas as pd

def generate_sensor_data(samples=1000, anomaly_ratio=0.05):
    np.random.seed(42)

    data = {
        "motion": np.random.normal(0.5, 0.1, samples),
        "proximity": np.random.normal(1.0, 0.2, samples),
        "noise": np.random.normal(30, 5, samples),
        "label": np.zeros(samples)
    }

    anomaly_count = int(samples * anomaly_ratio)
    anomaly_indices = np.random.choice(samples, anomaly_count, replace=False)

    data["motion"][anomaly_indices] += np.random.uniform(1, 2, anomaly_count)
    data["noise"][anomaly_indices] += np.random.uniform(20, 40, anomaly_count)
    data["label"][anomaly_indices] = 1

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_sensor_data()
    df.to_csv("data/sensor_data.csv", index=False)
    print("Sensor data generated.")
