# src/data_processing.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import FunctionTransformer

def create_rfm_features(df):
    # Convert TransactionStartTime to naive datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True).dt.tz_localize(None)
    current_date = datetime(2018, 11, 16)  # Naive datetime
    # Aggregate RFM features by CustomerId
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (current_date - pd.to_datetime(x.max())).days,  # Recency
        'TransactionId': 'count',  # Frequency
        'Value': 'sum'  # Monetary
    }).rename(columns={'TransactionStartTime': 'Recency', 'TransactionId': 'Frequency', 'Value': 'Monetary'})
    return rfm.reset_index()

def build_feature_pipeline():
    # Define columns based on original DataFrame (to be adjusted after RFM)
    categorical_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    numerical_cols = ['Amount', 'Value']

    # Preprocessing for numerical data
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Create pipeline
    pipeline = Pipeline(steps=[
        ('rfm', FunctionTransformer(create_rfm_features, validate=False))
    ])

    return pipeline

if __name__ == "__main__":
    import os
    os.chdir("C:/Users/hp/Desktop/Kifiya AIM/week 5/Technical Content/workspace")  # Set working directory
    data_path = "data/raw/transactions.csv"
    print("Attempting to load from:", data_path)
    df = pd.read_csv(data_path)
    print("Dataset columns:", df.columns.tolist())  # Debug column names
    pipeline = build_feature_pipeline()  # No need to pass df here
    X = pipeline.fit_transform(df)
    print("Feature matrix shape:", X.shape)  # Add this back
    os.makedirs('../data/processed', exist_ok=True)
    pd.DataFrame(X).to_csv('../data/processed/features.csv', index=False)
    print("Processed features saved to ../data/processed/features.csv")