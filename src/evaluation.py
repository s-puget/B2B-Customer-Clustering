import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_clustering(df):
    if "cluster" in df.columns and df["cluster"].nunique() > 1:
        features = df.drop(columns=["cluster", "client_id"])
        labels = df["cluster"]
        silhouette = silhouette_score(features, labels)
        davies_bouldin = davies_bouldin_score(features, labels)
        chi_score = calinski_harabasz_score(features, labels)
        return {"Silhouette Score": silhouette, "Davies-Bouldin Score": davies_bouldin, "Calinski Harabasz Score": chi_score}
    return {"Silhouette Score": None, "Davies-Bouldin Score": None, "Calinski Harabasz Score": None}

def run():
    df = pd.read_csv("clustered_data.csv")
    scores = evaluate_clustering(df)
    print(scores)    

if __name__ == "__main__":
    run()