import pandas as pd

def load_data(summary_path: str, transaction_path: str):
    df1 = pd.read_csv(summary_path)
    df2 = pd.read_csv(transaction_path)
    df = pd.merge(df1, df2, on="client_id", how="inner")
    return df

def run():
    df = load_data("data/client_summary_segmentation.csv", "data/client_transaction_product_file.csv")
    df.to_csv("merged_data.csv", index=False)

if __name__ == "__main__":
    run()