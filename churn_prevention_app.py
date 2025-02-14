"""
churn_prevention_app.py

Streamlit app that:
1. Loads churn_dataset.csv and churn_model.pkl
2. Scores each user for churn risk
3. Displays top high-risk users
4. Uses an OpenAI LLM to generate a "retention strategy" for each user
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI

# ----------------- OPENAI SETUP -------------------------------------------- #
# If you have your own specialized client, you can adapt accordingly.
# For a simple approach, pass your API key here:
client = OpenAI(
    api_key="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"
    # or: api_key=os.getenv("OPENAI_API_KEY")
)

def generate_retention_suggestion(user_info: dict, risk_score: float) -> str:
    """
    Calls an LLM to suggest a retention strategy for a given user at risk.
    user_info: dictionary of user fields
    risk_score: predicted churn risk
    """
    prompt = f"""
We have a user with ID {user_info['user_id']} who has a churn risk score of {risk_score:.2f}.
They have:
- {user_info['num_sessions']} sessions
- Average session length of {user_info['avg_session_length']} minutes
- Funnel completion {user_info['funnel_completion_pct']}%
- Total spend so far: {user_info['total_spend']}
Device preference: {user_info['orig_device_preference']}
Primary channel: {user_info['orig_primary_channel']}

Propose a short, personalized action or message to retain them.
Consider if a discount, personal outreach, or new feature highlight might help.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4" if you have it, or your custom "gpt-4o-mini"
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.8
        )
        suggestion = response.choices[0].message.content.strip()
        return suggestion
    except Exception as e:
        return f"Error calling LLM: {e}"

# ----------------- STREAMLIT APP ------------------------------------------- #
st.set_page_config(page_title="Churn Prevention App", layout="wide")

st.title("Churn Prevention & Retention Strategies")

st.write("""
This app scores users for churn risk using an XGBoost model, then uses an LLM 
to propose personalized retention strategies for high-risk users.
""")

# ------------------------------------------------------------------------- #
# 1) LOAD THE MODEL                                                         #
# ------------------------------------------------------------------------- #
model_tuple = joblib.load("churn_model.pkl")
model, feature_cols = model_tuple

# ------------------------------------------------------------------------- #
# 2) LOAD & PREP THE DATA                                                  #
# ------------------------------------------------------------------------- #
df = pd.read_csv("churn_dataset.csv")

# Convert dates
df["signup_date"] = pd.to_datetime(df["signup_date"])
df["last_active_date"] = pd.to_datetime(df["last_active_date"])

# Basic feature engineering
df["days_as_user"] = (df["last_active_date"] - df["signup_date"]).dt.days
df["days_since_signup"] = (pd.to_datetime("2023-07-01") - df["signup_date"]).dt.days

# 2a) Preserve original columns for LLM's text prompt
df["orig_device_preference"] = df["device_preference"]
df["orig_primary_channel"] = df["primary_channel"]

# 2b) Dummy-encode for the model
df = pd.get_dummies(df, columns=["device_preference", "primary_channel"], drop_first=True)

# We'll filter the dataset to those who have not actually churned yet
df_active = df[df["churned"] == 0].copy()

# Prepare data for scoring
X_active = df_active[feature_cols]

# Predict churn risk
df_active["churn_risk_score"] = model.predict_proba(X_active)[:, 1]

# ------------------------------------------------------------------------- #
# 3) UI CONTROLS                                                            #
# ------------------------------------------------------------------------- #
risk_threshold = st.slider("Risk Threshold (0 to 1)", 0.0, 1.0, 0.6, 0.01)
df_risky = df_active[df_active["churn_risk_score"] >= risk_threshold].copy()
df_risky = df_risky.sort_values("churn_risk_score", ascending=False)

st.write(f"Number of at-risk users above threshold {risk_threshold}: {df_risky.shape[0]}")

top_n = st.number_input("How many top risk users to display?", min_value=1, max_value=100, value=10)
df_top = df_risky.head(top_n).copy()

if df_top.empty:
    st.warning("No users found above that risk threshold. Lower it or generate a bigger dataset.")
else:
    st.subheader("High-Risk Users")
    st.write("Below are the top at-risk users. Click 'Generate Suggestions' to see LLM-based retention ideas.")

    # --------------------------------------------------------------------- #
    # 4) GENERATE LLM SUGGESTIONS                                           #
    # --------------------------------------------------------------------- #
    if st.button("Generate Suggestions for Top Users"):
        suggestions = []
        for idx, row in df_top.iterrows():
            user_dict = row.to_dict()
            suggestion = generate_retention_suggestion(user_dict, row["churn_risk_score"])
            suggestions.append(suggestion)

        df_top["suggested_action"] = suggestions

        st.write(df_top[[
            "user_id", 
            "churn_risk_score",
            "num_sessions",
            "funnel_completion_pct",
            "total_spend",
            "orig_device_preference",  # show original device for clarity
            "orig_primary_channel",    # show original channel for clarity
            "suggested_action"
        ]])
    else:
        st.info("Click 'Generate Suggestions' to see LLM-based retention strategies.")

st.write("---")
st.caption("End of Churn Prevention PoC.")
