"""
Unified Streamlit App using the new 'OpenAI' class approach
(where you do 'from openai import OpenAI' and create a client).
"""

import os
import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# IMPORTANT: Use the same library/import style as your working snippet:
from openai import OpenAI

###############################################################################
#                         CREATE YOUR OPENAI CLIENT                           #
###############################################################################
# Replace this with your actual API key (or manage via environment variable).
# If you prefer environment variables, do:
#    api_key = os.getenv("OPENAI_API_KEY")
# and then pass it below.
client = OpenAI(
    api_key="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"
)

###############################################################################
#                          DATA GENERATION                                    #
###############################################################################
np.random.seed(42)

weeks = ["Week 1", "Week 2", "Week 3"]
age_groups = ["Gen Z", "Millennial", "Gen X", "Boomer"]
funnel_stages = ["Landing Page", "Sign-Up", "Onboarding", "Purchase"]

data_records = []
for week in weeks:
    for age in age_groups:
        landing = np.random.randint(500, 1000)
        signup = int(landing * np.random.uniform(0.4, 0.8))
        onboarding = int(signup * np.random.uniform(0.6, 0.9))
        purchase = int(onboarding * np.random.uniform(0.5, 0.8))
        
        data_records.append({
            "Week": week,
            "Age Group": age,
            "Funnel Stage": "Landing Page",
            "Users": landing
        })
        data_records.append({
            "Week": week,
            "Age Group": age,
            "Funnel Stage": "Sign-Up",
            "Users": signup
        })
        data_records.append({
            "Week": week,
            "Age Group": age,
            "Funnel Stage": "Onboarding",
            "Users": onboarding
        })
        data_records.append({
            "Week": week,
            "Age Group": age,
            "Funnel Stage": "Purchase",
            "Users": purchase
        })

df_funnel = pd.DataFrame(data_records)

###############################################################################
#                         HELPER FUNCTION FOR LLM                             #
###############################################################################
def generate_openai_response(prompt: str) -> str:
    """
    Uses the client.chat.completions.create(...) interface 
    (the same approach that worked in your snippet with 'gpt-4o-mini')
    """
    # We build the 'messages' array just like Chat Completions requires:
    messages = [
        {"role": "system", "content": "You are a data analysis assistant."},
        {"role": "user",   "content": prompt}
    ]
    try:
        # If you have access to 'gpt-4o-mini', you can swap below:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4o-mini" if it's available in your plan
            store=True,            # 'store=True' is optional, depends on your library
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in OpenAI response: {e}"

###############################################################################
#                              STREAMLIT APP                                  #
###############################################################################
st.set_page_config(
    page_title="GenAI-Powered Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("GenAI-Powered Analytics Dashboard")
st.write(
    """
    This app demonstrates:
    1. **Funnel Analysis** for different user stages (Landing → Sign-Up → Onboarding → Purchase).
    2. **Pie Chart** for user segment distribution.
    3. **LLM-Generated Summaries** using the same approach as your working code snippet.
    4. **Scenario Planning** with a text input for "What if?" questions.
    """
)

# --------------------------- SIDEBAR CONTROLS ------------------------------- #
st.sidebar.header("Dashboard Controls")
selected_week = st.sidebar.selectbox("Select Week:", weeks, index=2)

scenario_question = st.sidebar.text_area(
    "AI Scenario Planning",
    placeholder="What if we move the sign-up form to the top?"
)
run_scenario = st.sidebar.button("Run Scenario Analysis")

# --------------------------- FUNNEL ANALYSIS -------------------------------- #
st.subheader(f"Funnel Analysis for {selected_week}")

dff_week = df_funnel[df_funnel["Week"] == selected_week]

aggregated = (
    dff_week.groupby("Funnel Stage")["Users"]
    .sum()
    .reset_index()
    .sort_values("Users", ascending=False)
)

fig_funnel = go.Figure(go.Funnel(
    y=aggregated["Funnel Stage"],
    x=aggregated["Users"],
    textinfo="value+percent previous"
))
fig_funnel.update_layout(
    title=f"Funnel Stages ({selected_week})",
    margin=dict(l=80, r=80, t=60, b=50),
    template="plotly_white"
)

st.plotly_chart(fig_funnel, use_container_width=True)

# Funnel narrative from LLM
funnel_prompt = f"""
We have the following funnel data for {selected_week}:
{aggregated.to_dict(orient='records')}

Please provide a concise interpretation or insight about potential reasons for changes in conversion.
"""
funnel_narrative = generate_openai_response(funnel_prompt)
st.info(f"**Funnel Analysis Narrative**: {funnel_narrative}")

# --------------------- PIE CHART: USER SEGMENTS ----------------------------- #
st.subheader(f"User Segment Distribution ({selected_week})")

pie_data = dff_week.groupby("Age Group")["Users"].sum().reset_index()
fig_pie = px.pie(
    pie_data,
    names="Age Group",
    values="Users",
    title=f"User Segments for {selected_week}",
    hole=0.3
)
fig_pie.update_layout(template="plotly_white")

st.plotly_chart(fig_pie, use_container_width=True)

# Pie chart narrative from LLM
pie_prompt = f"""
We have the following user segment distribution for {selected_week}:
{pie_data.to_dict(orient='records')}

Please provide insights about which segments are largest or if there's anything interesting or unusual.
"""
pie_narrative = generate_openai_response(pie_prompt)
st.info(f"**Segment Distribution Narrative**: {pie_narrative}")

# -------------------------- SCENARIO PLANNING ------------------------------ #
st.subheader("Scenario Planning / 'What-If' Analysis")

if run_scenario and scenario_question.strip():
    scenario_prompt = f"""
    We have historical data about sign-up flows, pricing, and user behaviors.
    Forecast or explain what might happen if:
    '{scenario_question}'

    Provide data-informed reasoning or disclaimers as appropriate.
    """
    scenario_response = generate_openai_response(scenario_prompt)
    st.success(f"**Scenario Analysis**:\n\n{scenario_response}")
elif run_scenario and not scenario_question.strip():
    st.warning("Please enter a scenario question before running the analysis.")

st.write("---")
st.caption("Built with Streamlit, Plotly, and the new 'OpenAI' client library.")
