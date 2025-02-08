import pandas as pd
import numpy as np

def weight_parameters(df, weights):
    for col, weight in weights.items():
        if col in df.columns:
            df[col] = df[col] * weight
    return df

if __name__ == "__main__":
    df = pd.read_csv("processed_data.csv")
    weights = {"feature1": 1.5, "feature2": 2.0, "feature3": 0.8}  # Adjust based on importance
    df = weight_parameters(df, weights)
    df.to_csv("weighted_data.csv", index=False)