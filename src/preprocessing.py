import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

def preprocess_data(df):

    # Drop unnecessary, redundant and low variance columns
    df = df.drop(columns=["Unnamed: 0", "most_buyed_product", "most_ordered_product_1", "most_ordered_product_2", "most_ordered_product_3", "favorite_prd_category_1", "favorite_prd_category_3", "favorite_prd_category_4", "agence_code", "brand_name", "nb_different_product_ordered", "purchase_activity_rate", "interaction_rate"], errors="ignore")
    df = df.rename(columns={"type": "client_type"})

    # Filling missing values
    for col in ['client_type', 'market']:
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)

    # Fix skewed columns
    skewed_cols = ["average_time_between_searches_days", "avg_products_seen", "nb_total_orders", "avg_quantity_per_order", "unique_prd_category_count"]
    for col in skewed_cols:
        df[col] = np.log1p(df[col])
        upper_percentile = df[col].quantile(0.95)
        df[col] = df[col].clip(upper=upper_percentile)


    # Encode categorical features
    one_hot_encoding_columns = ["device_usage_category", "client_type", "market", "preferred_channel"]
    other_encoding_columns = ["favorite_prd_category_2"]
    df_one_hot = pd.get_dummies(df[one_hot_encoding_columns], drop_first=True)
    ord_encoder = OrdinalEncoder()
    df_ordinal = df[other_encoding_columns].copy()
    df_ordinal[other_encoding_columns] = ord_encoder.fit_transform(df_ordinal)
    df = df.drop(columns=one_hot_encoding_columns + other_encoding_columns)
    df = pd.concat([df, df_one_hot, df_ordinal], axis=1)
    
    # Normalize numerical features
    feature_cols = df.drop('client_id', axis=1).columns
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df

if __name__ == "__main__":
    df = pd.read_csv("merged_data.csv")
    df = preprocess_data(df)
    df.to_csv("processed_data.csv", index=False)