import pandas as pd
from sklearn.cluster import DBSCAN

def apply_dbscan(df, eps=1.5):
    dbscan = DBSCAN(eps=eps)
    dbscan_labels = dbscan.fit_predict(df.drop('client_id', axis=1))
    df["cluster"] = dbscan_labels
    return df

def run():
    df = pd.read_csv("weighted_data.csv")
    df = apply_dbscan(df)
    df.to_csv("clustered_data.csv", index=False)

if __name__ == "__main__":
    run()