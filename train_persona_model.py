"""
train_persona_model.py

1. Reads persona_dataset.csv
2. Minimal feature engineering
3. Encodes or dummies for categorical columns
4. Trains K-Means to find user "personas"
5. Saves the model to 'persona_model.pkl'
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Convert signup_date to numeric if desired, or just ignore it.
    df["signup_date"] = pd.to_datetime(df["signup_date"])

    # We'll create a "days_since_signup" as a numeric feature for example
    reference_date = pd.to_datetime("2023-04-01")
    df["days_since_signup"] = (reference_date - df["signup_date"]).dt.days

    # One-hot encode device, primary_channel, funnel_stage_most_often
    df = pd.get_dummies(df, columns=["device", "primary_channel", "funnel_stage_most_often"], drop_first=True)
    return df

def get_feature_cols(df: pd.DataFrame):
    # We'll exclude user_id, signup_date from modeling
    # We'll keep days_since_signup, pages_visited, avg_time_on_site, purchase_count, total_spend, etc.
    exclude_cols = ["user_id", "signup_date"]
    return [c for c in df.columns if c not in exclude_cols]

if __name__ == "__main__":
    df = pd.read_csv("persona_dataset.csv")
    df = feature_engineering(df)

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].copy()

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means with e.g. 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_scaled)

    # Save model and additional info
    joblib.dump((kmeans, scaler, feature_cols), "persona_model.pkl")
    print("Saved persona_model.pkl with K-Means model.")
