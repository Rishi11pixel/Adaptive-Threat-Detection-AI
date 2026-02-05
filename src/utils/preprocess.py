import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/sensor_data.csv")

scaler = StandardScaler()
features = df.drop("label", axis=1)
scaled_features = scaler.fit_transform(features)

processed_df = pd.DataFrame(scaled_features, columns=features.columns)
processed_df["label"] = df["label"]

processed_df.to_csv("data/processed_sensor_data.csv", index=False)
print("Data preprocessed.")
