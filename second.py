#################################
# app.py (No LLM / OpenAI logic)
#################################

import os
import random
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

#################################
# 1. Generate Synthetic Data
#################################

def create_synthetic_user_data(num_users=1000, start_date="2025-01-01", end_date="2025-02-01"):
    """
    Generate random user journey data.
    Each row simulates a user session with various attributes.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = {
        "user_id": np.random.randint(1, num_users, size=len(date_range)*5),  # 5 sessions per day
        "session_date": np.random.choice(date_range, size=len(date_range)*5),
        "age_group": np.random.choice(
            ["18-24", "25-34", "35-44", "45-54", "55+"], 
            size=len(date_range)*5, 
            p=[0.2, 0.3, 0.25, 0.15, 0.1]
        ),
        "channel": np.random.choice(
            ["organic", "paid_search", "email", "social", "referral"], 
            size=len(date_range)*5
        ),
        "funnel_stage": np.random.choice(
            ["awareness", "consideration", "conversion", "retention"], 
            size=len(date_range)*5, 
            p=[0.4, 0.3, 0.2, 0.1]
        ),
        "spent_time_sec": np.random.gamma(shape=2, scale=30, size=len(date_range)*5),  # random time on site
        "conversion_value": np.random.choice(
            [0, 10, 20, 50, 100], 
            size=len(date_range)*5, 
            p=[0.75, 0.1, 0.05, 0.05, 0.05]
        )
    }
    
    df = pd.DataFrame(data)
    df["session_date"] = pd.to_datetime(df["session_date"])
    return df

#################################
# 2. Streamlit App
#################################

def main():
    st.set_page_config(page_title="Analytics Dashboard (No LLM)", layout="wide")
    st.title("Analytics Dashboard (No LLM)")
    st.write("""
    **Explore user journey data without any AI/LLM logic.**
    """)

    # Generate or load data
    st.sidebar.header("Data Options")
    num_users = st.sidebar.slider("Number of synthetic users", 500, 5000, 1000, step=500)
    df = create_synthetic_user_data(num_users=num_users)

    # --- SECTION A: Data Filtering ---
    st.subheader("1. Data Filtering")
    unique_channels = df["channel"].unique()
    selected_channels = st.multiselect(
        "Select channels:", 
        options=unique_channels, 
        default=list(unique_channels)
    )

    unique_ages = df["age_group"].unique()
    selected_age_groups = st.multiselect(
        "Select age groups:", 
        options=unique_ages, 
        default=list(unique_ages)
    )

    # Filter the DataFrame
    filtered_df = df[
        (df["channel"].isin(selected_channels)) &
        (df["age_group"].isin(selected_age_groups))
    ]
    st.write(f"Showing {len(filtered_df)} sessions after filtering.")

    # --- SECTION B: Visualization ---
    st.subheader("2. Visualizations & Click-to-Query")

    # Funnel Stage Breakdown (Pie Chart)
    funnel_counts = filtered_df["funnel_stage"].value_counts().reset_index()
    funnel_counts.columns = ["funnel_stage", "count"]

    fig_funnel = px.pie(
        funnel_counts, 
        names="funnel_stage", 
        values="count", 
        title="Funnel Stage Distribution (Filtered)"
    )
    fig_funnel.update_traces(hoverinfo='label+percent', textinfo='value')
    
    st.plotly_chart(fig_funnel, use_container_width=True)
    st.caption("Example funnel stage breakdown. (No direct click-to-query logic here.)")

    # Simple text input to mimic a "query by stage"
    selected_stage = st.text_input(
        "Type a funnel stage to see a breakdown (e.g., 'awareness', 'conversion'):"
    )
    if selected_stage:
        deeper_df = filtered_df[filtered_df["funnel_stage"] == selected_stage]
        st.markdown(f"**You selected funnel stage:** {selected_stage}")
        st.write(f"Total sessions in this stage: {len(deeper_df)}")

        # Deeper stats
        avg_time = deeper_df["spent_time_sec"].mean()
        conv_rate = (deeper_df["conversion_value"] > 0).mean()
        st.write(f"Average time spent: {avg_time:.2f} sec")
        st.write(f"Conversion rate (subset): {conv_rate:.2%}")

    # Conversion Value Over Time (Line Chart)
    st.subheader("Conversion Value Over Time")
    daily_revenue = filtered_df.groupby("session_date")["conversion_value"].sum().reset_index()
    fig_time = px.line(
        daily_revenue, 
        x="session_date", 
        y="conversion_value", 
        title="Daily Conversion Value"
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # --- SECTION C: Scenario Simulation (Placeholder) ---
    st.subheader("3. Scenario Simulation (Placeholder)")
    st.write("""
    Normally, we'd call an AI/LLM to generate "what if" insights, 
    but in this no-LLM version, we just display a placeholder.
    """)
    scenario_text = st.text_input("Propose a hypothetical scenario:")
    if st.button("Run Scenario Simulation"):
        if scenario_text:
            st.info(f"This is where the AI insight would appear for: '{scenario_text}'")
        else:
            st.warning("Please enter a scenario before running the simulation.")

    # End of Streamlit app
    st.write("---")
    st.write("**Note:** This is a purely data-driven example, without any OpenAI or LLM calls.")


if __name__ == "__main__":
    main()
