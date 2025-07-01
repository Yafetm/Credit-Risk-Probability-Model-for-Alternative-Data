# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def train_model():
    # Load processed features and target with absolute path
    data_path = "C:/Users/hp/Desktop/Kifiya AIM/week 5/Technical Content/workspace/data/processed/features.csv"
    raw_path = "C:/Users/hp/Desktop/Kifiya AIM/week 5/Technical Content/workspace/data/raw/transactions.csv"
    print("Loading from:", data_path, "and", raw_path)
    X = pd.read_csv(data_path).set_index('CustomerId')
    df = pd.read_csv(raw_path)
    # Aggregate FraudResult by mode (most frequent value) per CustomerId
    y = df.groupby('CustomerId')['FraudResult'].agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
    X = X.loc[y.index]  # Align X with y

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X[['Recency', 'Frequency', 'Monetary']], y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Model Performance:\n", classification_report(y_test, y_pred))

    # Save model
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/rf_model.pkl')
    print("Model saved to ../models/rf_model.pkl")

if __name__ == "__main__":
    train_model()