"""
train_churn_model.py

Trains an XGBoost model to predict 'churned' from the synthetic dataset.
Saves the model to 'churn_model.pkl'.
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal feature engineering. Convert dates to numeric,
    encode categorical, etc.
    """
    # Convert dates to numeric features: days since signup, days since last active
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["last_active_date"] = pd.to_datetime(df["last_active_date"])

    # We'll pick an arbitrary "reference date" for numeric conversion
    # or we can just measure difference between last_active and signup
    df["days_as_user"] = (df["last_active_date"] - df["signup_date"]).dt.days
    df["days_since_signup"] = (pd.to_datetime("2023-07-01") - df["signup_date"]).dt.days

    # Encode device_preference, primary_channel
    df = pd.get_dummies(df, columns=["device_preference", "primary_channel"], drop_first=True)

    return df

def get_feature_cols(df: pd.DataFrame):
    # We'll exclude user_id, dates, churned (target)
    exclude_cols = ["user_id", "signup_date", "last_active_date", "churned"]
    return [c for c in df.columns if c not in exclude_cols]

if __name__ == "__main__":
    df = pd.read_csv("churn_dataset.csv")
    df = feature_engineering(df)

    # Prepare features X, target y
    feature_cols = get_feature_cols(df)
    X = df[feature_cols]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=y)

    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {auc_score:.3f}")

    # Save model
    joblib.dump((model, feature_cols), "churn_model.pkl")
    print("Saved churn_model.pkl.")
