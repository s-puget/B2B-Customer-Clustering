import pandas as pd

def weight_parameters(df):
    df["nb_total_orders"] = df["nb_total_orders"] * 3
    df.loc[:, (df.nunique() == 2)] *= 3
    return df

def run():
    df = pd.read_csv("processed_data.csv")
    df = weight_parameters(df)
    df.to_csv("weighted_data.csv", index=False)

if __name__ == "__main__":
    run()