import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.rename(columns={"type": "client_type"})
    
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=["object"]).columns
    encoder = OrdinalEncoder()
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=["number"]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

if __name__ == "__main__":
    df = pd.read_csv("merged_data.csv")
    df = preprocess_data(df)
    df.to_csv("processed_data.csv", index=False)