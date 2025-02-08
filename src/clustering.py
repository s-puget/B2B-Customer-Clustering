import pandas as pd
from sklearn.cluster import DBSCAN

def apply_dbscan(df, eps=0.5, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    df["cluster"] = clustering.fit_predict(df)
    return df

if __name__ == "__main__":
    df = pd.read_csv("weighted_data.csv")
    df = apply_dbscan(df, eps=0.3, min_samples=10)  # Tune hyperparameters as needed
    df.to_csv("clustered_data.csv", index=False)
    df.to_csv("clustered_data.csv", index=False)