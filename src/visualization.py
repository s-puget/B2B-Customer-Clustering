import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def PCA_visualization(df): 

    # Normalize numerical features
    feature_cols = df.drop('client_id', axis=1).columns
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df.drop(["client_id", "cluster"], axis=1))
    
    df_pca = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
    df_pca["Cluster"] = df["cluster"]

    # Get unique clusters and create a more contrasted colormap
    unique_clusters = np.unique(df_pca["Cluster"])
    num_clusters = len(unique_clusters)
    
    # Use "rainbow" colormap for higher contrast
    colors = plt.cm.get_cmap("rainbow", num_clusters)

    plt.figure(figsize=(8, 6))
    
    for i, cluster in enumerate(unique_clusters):
        cluster_data = df_pca[df_pca["Cluster"] == cluster]
        plt.scatter(cluster_data["PCA1"], cluster_data["PCA2"], color=colors(i), s=10)

    plt.title("DBSACN Clustering (PCA-reduced dimensions)")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.xticks([])
    plt.yticks([])
    plt.show()

def run():
    df = pd.read_csv("clustered_data.csv")
    PCA_visualization(df)

if __name__ == "__main__":
    run()