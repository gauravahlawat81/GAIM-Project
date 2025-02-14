#################################
# app.py
#################################
import os
import random
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# If you want environment variable loading:
# from dotenv import load_dotenv
# load_dotenv()

import openai  # or any LLM library you prefer

#################################
# 1. Configuration
#################################

# Set your OpenAI API key (or other LLM credentials)
# Option 1: Load from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Option 2: Hardcode for demo ONLY (not recommended for production)
# openai.api_key = "sk-1234..."


# Helper function to call the OpenAI LLM (GPT-like model)
def generate_narrative(prompt: str, max_tokens=100) -> str:
   client = openai.OpenAI()  # NEW client object for OpenAI v1.0+
   response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0
    )
   text = response.choices[0].message.content.strip()
   return text

#################################
# 2. Generate Synthetic Data
#################################

def create_synthetic_user_data(num_users=1000, start_date="2025-01-01", end_date="2025-02-01"):
    """
    Generate random user journey data.
    Each row simulates a user session with various attributes.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = {
        "user_id": np.random.randint(1, num_users, size=len(date_range)*5),  # 5 sessions per day on average
        "session_date": np.random.choice(date_range, size=len(date_range)*5),
        "age_group": np.random.choice(["18-24", "25-34", "35-44", "45-54", "55+"], size=len(date_range)*5, p=[0.2, 0.3, 0.25, 0.15, 0.1]),
        "channel": np.random.choice(["organic", "paid_search", "email", "social", "referral"], size=len(date_range)*5),
        "funnel_stage": np.random.choice(["awareness", "consideration", "conversion", "retention"], size=len(date_range)*5, p=[0.4, 0.3, 0.2, 0.1]),
        "spent_time_sec": np.random.gamma(shape=2, scale=30, size=len(date_range)*5),  # random time on site
        "conversion_value": np.random.choice([0, 10, 20, 50, 100], size=len(date_range)*5, p=[0.75, 0.1, 0.05, 0.05, 0.05])
    }
    
    df = pd.DataFrame(data)
    df["session_date"] = pd.to_datetime(df["session_date"])
    return df


#################################
# 3. Scenario Simulation
#################################
def scenario_simulation(scenario_description: str, df: pd.DataFrame) -> str:
    """
    Simple 'what if' scenario logic:
    - We will feed some aggregated data + scenario to the LLM
    - LLM returns a forecast or narrative

    In a real case, you might integrate a regression model or
    marketing mix model to produce more precise predictions.
    """
    # Summarize some basic current metrics
    total_sessions = len(df)
    conversion_rate = (df["conversion_value"] > 0).mean()
    avg_conversion_val = df["conversion_value"].mean()
    
    prompt = f"""
    We have a website with {total_sessions} recent sessions. 
    The current conversion rate is {conversion_rate:.2%} 
    and the average revenue per session is ${avg_conversion_val:.2f}.
    
    Scenario: {scenario_description}

    Based on this scenario, provide a short forecast or 
    recommendation on how metrics (conversion rate, revenue, user engagement) 
    might change. Use hypothetical, data-inspired reasoning.
    """

    return generate_narrative(prompt, max_tokens=150)


#################################
# 4. Streamlit App
#################################

def main():
    st.set_page_config(page_title="GenAI-Powered Analytics Dashboard", layout="wide")
    st.title("GenAI-Powered Analytics Dashboard")
    st.write("""
    **Explore user journey data, generate AI-driven insights, and run scenario simulations.**
    """)

    # Generate or load data
    st.sidebar.header("Data Options")
    num_users = st.sidebar.slider("Number of synthetic users", 500, 5000, 1000, step=500)
    df = create_synthetic_user_data(num_users=num_users)

    # -- SECTION A: Data Filtering --
    st.subheader("1. Data Filtering")
    unique_channels = df["channel"].unique()
    selected_channels = st.multiselect("Select channels:", options=unique_channels, default=list(unique_channels))

    unique_ages = df["age_group"].unique()
    selected_age_groups = st.multiselect("Select age groups:", options=unique_ages, default=list(unique_ages))

    # Filter the dataframe
    filtered_df = df[
        (df["channel"].isin(selected_channels)) &
        (df["age_group"].isin(selected_age_groups))
    ]

    st.write(f"Showing {len(filtered_df)} sessions after filtering.")

    # -- SECTION B: Visualization --
    st.subheader("2. Visualizations & Click-to-Query")
    
    # Example 1: Funnel Stage Breakdown
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
    st.caption("Click on a slice in this chart for deeper insights. (Streamlit only partially supports Plotly events.)")

    # Pseudo “click event” logic:
    # In a real scenario, you'd capture selection events with Plotly callbacks (in Dash) or advanced Streamlit customization.
    # For now, let's just let the user type the funnel stage they want to query.
    selected_stage = st.text_input("Type a funnel stage to query deeper insights (e.g., 'awareness', 'conversion'):")
    if selected_stage:
        # Filter by funnel stage
        deeper_df = filtered_df[filtered_df["funnel_stage"] == selected_stage]
        st.markdown(f"**You selected funnel stage:** {selected_stage}")
        st.write(f"Total sessions in this stage: {len(deeper_df)}")

        # (Optional) Generate some deeper stats
        avg_time = deeper_df["spent_time_sec"].mean()
        conv_rate = (deeper_df["conversion_value"] > 0).mean()
        st.write(f"Average time spent: {avg_time:.2f} sec")
        st.write(f"Conversion rate (within this subset): {conv_rate:.2%}")

        # Generate a narrative using an LLM
        deeper_prompt = f"""
        We have {len(deeper_df)} sessions in the funnel stage '{selected_stage}'. 
        The average time spent is {avg_time:.1f} seconds, 
        and the conversion rate is {conv_rate:.2%} in this subset.
        
        Please provide a brief insight or recommendation 
        as to why the conversion rate might be at this level and 
        how we could optimize it.
        """
        deeper_narrative = generate_narrative(deeper_prompt, max_tokens=100)
        st.write("**AI-Generated Insight:**")
        st.info(deeper_narrative)

    # Example 2: Conversion Over Time
    st.subheader("Conversion Value Over Time")
    daily_revenue = filtered_df.groupby("session_date")["conversion_value"].sum().reset_index()
    fig_time = px.line(daily_revenue, x="session_date", y="conversion_value", title="Daily Conversion Value")
    st.plotly_chart(fig_time, use_container_width=True)

    # -- SECTION C: Predictive Highlights (Quick LLM Narrative) --
    st.subheader("3. Predictive Highlights (LLM Narrative)")
    # Summarize the filtered_df
    total_sessions = len(filtered_df)
    conversion_rate = (filtered_df["conversion_value"] > 0).mean()
    avg_conversion_val = filtered_df["conversion_value"].mean()

    predictive_prompt = f"""
    We have {total_sessions} sessions after applying the filters. 
    The current conversion rate is {conversion_rate:.2%}, 
    and the average conversion value is ${avg_conversion_val:.2f} per session.
    Identify any potential at-risk segments or emerging opportunities 
    based on these filtered data stats.
    """
    predictive_narrative = generate_narrative(predictive_prompt, max_tokens=150)
    st.write("**AI-Generated Predictive Highlight:**")
    st.success(predictive_narrative)

    # -- SECTION D: Scenario Simulation --
    st.subheader("4. Scenario Simulation")
    st.write("Ask a hypothetical question or propose a change, and let the AI produce a forecast.")
    scenario_text = st.text_input("Describe your scenario (e.g., 'Move sign-up form to top of page').")
    if st.button("Run Scenario Simulation"):
        if scenario_text:
            result = scenario_simulation(scenario_text, filtered_df)
            st.info(result)
        else:
            st.warning("Please enter a scenario before running the simulation.")

    # End of Streamlit app
    st.write("---")
    st.write("**Note:** This is a demo. In a production setup, incorporate real data pipelines, robust error handling, and secure LLM usage.")
    

if __name__ == "__main__":
    main()
