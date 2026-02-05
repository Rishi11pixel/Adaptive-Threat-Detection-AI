import numpy as np
import pandas as pd

def build_sequences(data, window_size=10):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
    return np.array(sequences)

df = pd.read_csv("data/processed_sensor_data.csv")
features = df.drop("label", axis=1).values

sequences = build_sequences(features)
np.save("data/sequences.npy", sequences)

print("Time-series sequences created.")
