"""
persona_app.py

Streamlit app that:
1. Loads persona_dataset.csv and persona_model.pkl
2. Assigns each user to a cluster (persona)
3. Uses an OpenAI LLM to generate a persona description
4. Displays summary & potential improvements for each persona
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# We'll use the same approach you use for OpenAI client, e.g. from openai import OpenAI
from openai import OpenAI

###############################################################################
#                         OPENAI SETUP                                        #
###############################################################################
client = OpenAI(
    api_key="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"
    # or: api_key=os.getenv("OPENAI_API_KEY")
)

def generate_persona_summary(cluster_id: int, user_stats: dict) -> str:
    """
    Calls the LLM to interpret a persona cluster's characteristics.
    user_stats: aggregated stats about that cluster
    """
    prompt = f"""
We have identified a user persona labeled 'Cluster {cluster_id}' with the following characteristics:
- Average pages visited: {user_stats["pages_visited"]:.1f}
- Average time on site (s): {user_stats["avg_time_on_site"]:.1f}
- Average purchase count: {user_stats["purchase_count"]:.1f}
- Average total spend: {user_stats["total_spend"]:.1f}
- Common devices: {user_stats["top_devices"]}
- Common channels: {user_stats["top_channels"]}
- Common funnel stages: {user_stats["top_stages"]}

Please provide a concise, user-friendly name for this persona 
(e.g., "Bargain Hunter," "Impulse Buyer") 
and explain why they behave this way 
and how we can tailor site layout or messaging to them.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.7
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        return f"Error calling LLM: {e}"

###############################################################################
#                           STREAMLIT APP                                     #
###############################################################################
st.set_page_config(page_title="Persona & Segmentation", layout="wide")

st.title("Automated User Journey Persona & Segmentation")

st.write("""
**Goal**: Cluster users into distinct personas based on their journey 
(pages visited, time spent, purchase patterns, etc.) 
and let an LLM summarize each persona in natural language.
""")

# 1) LOAD MODEL
model_tuple = joblib.load("persona_model.pkl")
kmeans, scaler, feature_cols = model_tuple

# 2) LOAD DATA
df_raw = pd.read_csv("persona_dataset.csv")

# We'll replicate the feature engineering from training
def feature_engineering_inference(df: pd.DataFrame):
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    reference_date = pd.to_datetime("2023-04-01")
    df["days_since_signup"] = (reference_date - df["signup_date"]).dt.days
    df = pd.get_dummies(df, columns=["device", "primary_channel", "funnel_stage_most_often"], drop_first=True)
    return df

df_inf = feature_engineering_inference(df_raw.copy())

# 3) Predict cluster for each user
X_inf = df_inf[feature_cols].copy()
X_scaled_inf = scaler.transform(X_inf)
df_raw["cluster_id"] = kmeans.predict(X_scaled_inf)

# 4) Let user pick how many random sample users to see, or see all
sample_count = st.slider("How many random users to preview?", 1, 50, 10)
df_sample = df_raw.sample(sample_count, random_state=42)

st.subheader("Sample of Assigned Clusters")
st.dataframe(df_sample[["user_id", "pages_visited", "avg_time_on_site", 
                        "purchase_count", "total_spend", 
                        "device", "primary_channel", 
                        "funnel_stage_most_often", "cluster_id"]])

# 5) Summarize each cluster persona with an LLM
st.subheader("Persona Summaries")
unique_clusters = sorted(df_raw["cluster_id"].unique())
if st.button("Generate Persona Summaries"):
    cluster_summaries = {}
    for cid in unique_clusters:
        # 5a) Calculate aggregated stats for users in this cluster
        subset = df_raw[df_raw["cluster_id"] == cid]
        pages_visited_avg = subset["pages_visited"].mean()
        time_on_site_avg = subset["avg_time_on_site"].mean()
        purchase_avg = subset["purchase_count"].mean()
        spend_avg = subset["total_spend"].mean()

        # Common device / channel / funnel usage (top 2 for each)
        top_devices = subset["device"].value_counts().nlargest(2).index.tolist()
        top_channels = subset["primary_channel"].value_counts().nlargest(2).index.tolist()
        top_stages = subset["funnel_stage_most_often"].value_counts().nlargest(2).index.tolist()

        # Build a dictionary of aggregated stats
        stats = {
            "pages_visited": pages_visited_avg,
            "avg_time_on_site": time_on_site_avg,
            "purchase_count": purchase_avg,
            "total_spend": spend_avg,
            "top_devices": top_devices,
            "top_channels": top_channels,
            "top_stages": top_stages,
        }

        persona_desc = generate_persona_summary(cid, stats)
        cluster_summaries[cid] = persona_desc

    # Display results
    for cid in unique_clusters:
        st.markdown(f"### Persona for Cluster {cid}")
        st.write(cluster_summaries[cid])
        st.write("---")

else:
    st.info("Click the 'Generate Persona Summaries' button to see LLM-based cluster descriptions.")

st.write("---")
st.caption("End of Persona Segmentation PoC.")
