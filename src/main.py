import data_loader
import preprocessing
import feature_engineering
import clustering
import evaluation
import visualization

def main():

    print("Loading data...")
    data_loader.run()

    print("Preprocessing data...")
    preprocessing.run()

    print("Applying feature engineering...")
    feature_engineering.run()

    print("Training model...")
    clustering.run()

    print("Clustering complete.\n")

    print("Clustering scores:")
    evaluation.run()
    print("\n")

    print("Excecuting PCA for visualization...")
    visualization.run()

    print("Visualization complete.\n")


if __name__ == "__main__":
    main()