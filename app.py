import os
import random
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Use the same library as your working snippet:
from openai import OpenAI

###############################################################################
#                         CREATE YOUR OPENAI CLIENT                           #
###############################################################################
client = OpenAI(
    api_key="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"
    # If you'd like to read from environment variables instead:
    # api_key = os.getenv("OPENAI_API_KEY")
)

###############################################################################
#                       LLM HELPER FUNCTION                                   #
###############################################################################
def generate_llm_response(prompt: str, model_name="gpt-4o-mini") -> str:
    """
    Generic function to call your GPT-based model 
    for chat completion using the 'OpenAI' client approach.
    """
    messages = [
        {"role": "system", "content": "You are an advanced data analysis assistant."},
        {"role": "user", "content": prompt}
    ]
    try:
        completion = client.chat.completions.create(
            model=model_name,
            store=True,
            messages=messages,
            max_tokens=3000,
            temperature=0.7
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in LLM response: {e}"

###############################################################################
#                       LOAD BIG DATA FROM CSV                                #
###############################################################################
@st.cache_data
def load_data(csv_path="complex_user_data.csv"):
    df = pd.read_csv(csv_path)
    # Convert event_date to datetime for timeseries analysis
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df

###############################################################################
#                           BUILD STREAMLIT APP                               #
###############################################################################
st.set_page_config(
    page_title="Advanced Multi-Channel Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Advanced Multi-Channel Analytics Dashboard")
st.write("""
**Goal**: Understand user journeys across channels, identify drop-off points, 
and see how friction leads to lost conversions and loyalty.
""")

# ----------------- DATA LOADING -------------------------------------------- #
df = load_data()

st.sidebar.header("Filters & Controls")

# 1) Date Range Filter
min_date = df["event_date"].min().date()
max_date = df["event_date"].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    (min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# 2) Channel Filter
channels = df["channel"].unique().tolist()
selected_channels = st.sidebar.multiselect("Channels", channels, default=channels)

# 3) Device Filter
devices = df["device"].unique().tolist()
selected_devices = st.sidebar.multiselect("Devices", devices, default=devices)

# 4) Scenario Text
scenario_text = st.sidebar.text_area(
    "Scenario Planning",
    placeholder="What if we increase our social media ad spend by 20%?"
)
scenario_run = st.sidebar.button("Run Scenario")

# Filter data based on user selection
start_filter, end_filter = date_range if isinstance(date_range, tuple) else (date_range, date_range)
dff = df[
    (df["event_date"] >= pd.to_datetime(start_filter)) &
    (df["event_date"] <= pd.to_datetime(end_filter)) &
    (df["channel"].isin(selected_channels)) &
    (df["device"].isin(selected_devices))
]

st.write(f"**Loaded Data**: {dff.shape[0]} events from {start_filter} to {end_filter}")
st.dataframe(dff.head(5))

# --------------------------------------------------------------------------- #
# 1) IMPROVED FUNNEL / SANKEY DIAGRAM                                         #
# --------------------------------------------------------------------------- #
st.subheader("Sankey Diagram of Funnel Stages")
st.write("""
A Sankey chart can show how many users move from one funnel stage 
to another, highlighting major drop-offs or progressions.
""")

# Sort dff by user_id, event_date, funnel_stage in a logical order:
dff_sorted = dff.sort_values(["user_id", "event_date", "funnel_stage"])
# Create a "next_stage" for each row based on the next row of the same user.
dff_sorted["next_stage"] = dff_sorted.groupby("user_id")["funnel_stage"].shift(-1)

# Filter out transitions that are NaN or the same stage
dff_transitions = dff_sorted.dropna(subset=["next_stage"])
dff_transitions = dff_transitions[dff_transitions["funnel_stage"] != dff_transitions["next_stage"]]

# Count transitions
transition_counts = dff_transitions.groupby(["funnel_stage", "next_stage"]).size().reset_index(name="count")

if transition_counts.empty:
    st.warning("No transitions found for the current filters. Adjust filters to see a Sankey.")
else:
    # Unique funnel stages
    all_stages = list(set(transition_counts["funnel_stage"].unique()).union(transition_counts["next_stage"].unique()))
    # Map stage to index
    stage_index_map = {stage: i for i, stage in enumerate(all_stages)}
    
    # Optional: Assign each stage a distinct color
    # For demonstration, define a color palette. 
    # (If there are more stages than colors, you'll need a bigger palette.)
    color_palette = [
        "#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", 
        "#EECA3B", "#B279A2", "#FF9DA7", "#9C755F", "#BAB0AC"
    ]
    # Create a color mapping with fallback if stages > len(color_palette)
    stage_colors = []
    for i, stg in enumerate(all_stages):
        stage_colors.append(color_palette[i % len(color_palette)])
    
    sankey_node = dict(
        label=all_stages,
        pad=30,
        thickness=20,
        color=stage_colors,
        hovertemplate='%{label}<extra></extra>'
    )
    sankey_link = dict(
        source=transition_counts["funnel_stage"].map(stage_index_map),
        target=transition_counts["next_stage"].map(stage_index_map),
        value=transition_counts["count"],
        color="rgba(0,0,0,0.2)"  # link color w/ slight transparency
    )
    
    fig_sankey = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=sankey_node,
        link=sankey_link
    )])
    
    fig_sankey.update_layout(
        title_text="Funnel Stage Transitions (Sankey)",
        font=dict(size=14, color="#000000"),
        width=1100,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(fig_sankey, use_container_width=True)
    
    sankey_prompt = f"""
We generated a Sankey diagram for funnel transitions with these transitions:
{transition_counts.to_dict(orient='records')}

Explain major user drop-off points or interesting observations in funnel transitions.
"""
    sankey_narrative = generate_llm_response(sankey_prompt)
    st.info(f"**Sankey Analysis Narrative**: {sankey_narrative}")


# --------------------------------------------------------------------------- #
# 1a) STAGE TRANSITION HEATMAP (Alternative to Sankey)                        #
# --------------------------------------------------------------------------- #
st.subheader("Stage Transition Heatmap")
st.write("""
A simpler matrix showing how many users go from each stage (rows) to the next (columns).
Useful if the Sankey becomes too busy.
""")

if transition_counts.empty:
    st.warning("No data for stage transition heatmap. Adjust filters.")
else:
    # Build pivot table (row=funnel_stage, col=next_stage)
    unique_stages = sorted(
        list(set(transition_counts["funnel_stage"]) | set(transition_counts["next_stage"]))
    )
    transition_matrix = transition_counts.pivot_table(
        index="funnel_stage",
        columns="next_stage",
        values="count",
        aggfunc="sum"
    ).fillna(0).reindex(index=unique_stages, columns=unique_stages, fill_value=0)
    
    fig_heatmap = px.imshow(
        transition_matrix,
        labels=dict(x="Next Stage", y="From Stage", color="User Count"),
        x=transition_matrix.columns,
        y=transition_matrix.index,
        color_continuous_scale="Blues",
        title="Stage-to-Stage Transition Heatmap"
    )
    fig_heatmap.update_xaxes(side="top")
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    heatmap_prompt = f"""
We have this transition matrix (rows = from stage, columns = next stage):
{transition_matrix.to_dict()}

Identify any stages with the highest drop-offs or unusual transitions, 
and suggest potential improvements.
"""
    heatmap_narrative = generate_llm_response(heatmap_prompt)
    st.info(f"**Heatmap Analysis Narrative**: {heatmap_narrative}")


# --------------------------------------------------------------------------- #
# 2) CHANNEL BREAKDOWN: Where do we see largest drops or purchases?
# --------------------------------------------------------------------------- #
st.subheader("Channel Breakdown: Drop-off vs. Purchase")
st.write("""
Compare each channel's share of user outcomes. 
Identify which channels lead to higher drop-offs and which channels lead to more purchases.
""")

channel_outcome = dff.groupby(["channel", "action_outcome"]).size().reset_index(name="count")
if channel_outcome.empty:
    st.warning("No data to display for channel breakdown. Adjust filters.")
else:
    fig_channel = px.bar(
        channel_outcome,
        x="channel",
        y="count",
        color="action_outcome",
        title="Channel vs. Outcome",
        barmode="group"
    )
    st.plotly_chart(fig_channel, use_container_width=True)
    
    channel_prompt = f"""
Channel breakdown (drop-off vs. purchase) is:
{channel_outcome.to_dict(orient='records')}

Please provide insights on which channels see the most drop-offs vs. purchases, 
and potential reasons why that might be happening.
"""
    channel_narrative = generate_llm_response(channel_prompt)
    st.info(f"**Channel Outcome Narrative**: {channel_narrative}")


# --------------------------------------------------------------------------- #
# 2a) MULTI-LEVEL FUNNEL BY CHANNEL                                          #
# --------------------------------------------------------------------------- #
st.subheader("Multi-Level Funnel (by Channel)")
st.write("""
A grouped bar chart showing how many users reach each funnel stage, segmented by channel.
Helps visualize stage-by-stage drop-offs across different channels.
""")

if "funnel_stage" in dff.columns:
    # For each channel & funnel stage, count how many events occurred
    funnel_counts = dff.groupby(["channel", "funnel_stage"]).size().reset_index(name="count")
    # Define an ordered list of stages, if you know your typical funnel order:
    ordered_stages = ["Landing Page", "Product View", "Add to Cart", "Checkout", "Payment", "Confirmation"]
    
    # Ensure funnel_stage is categorical to preserve order in the plot
    funnel_counts["funnel_stage"] = pd.Categorical(funnel_counts["funnel_stage"], categories=ordered_stages, ordered=True)
    funnel_counts = funnel_counts.dropna(subset=["funnel_stage"]).sort_values(by="funnel_stage")
    
    if funnel_counts.empty:
        st.warning("No funnel stage data available for multi-level funnel. Adjust filters.")
    else:
        fig_multi_funnel = px.bar(
            funnel_counts,
            x="funnel_stage",
            y="count",
            color="channel",
            title="Funnel Stages by Channel",
            barmode="group"
        )
        st.plotly_chart(fig_multi_funnel, use_container_width=True)
        
        prompt_funnel = f"""
We have a multi-level funnel grouped by channel. Data:
{funnel_counts.to_dict(orient='records')}

Highlight any channels or stages where user drop-offs are significantly higher.
"""
        analysis_funnel = generate_llm_response(prompt_funnel)
        st.info(f"**Channel-Level Funnel Narrative**: {analysis_funnel}")
else:
    st.warning("No 'funnel_stage' column in data. Cannot display multi-level funnel.")


# --------------------------------------------------------------------------- #
# 3) DAILY TREND: Are we improving or worsening over time?
# --------------------------------------------------------------------------- #
st.subheader("Time Series: Daily Purchase vs. Drop-off")
st.write("""
See if user outcomes are improving over time or if there are spikes in drop-offs.
""")

daily_outcomes = dff.groupby(["event_date", "action_outcome"]).size().reset_index(name="count")
if daily_outcomes.empty:
    st.warning("No data to display for daily trend. Adjust filters.")
else:
    fig_daily = px.line(
        daily_outcomes,
        x="event_date",
        y="count",
        color="action_outcome",
        title="Daily Outcome Trends"
    )
    st.plotly_chart(fig_daily, use_container_width=True)
    
    daily_prompt = f"""
We have the following daily outcome data for the selected range:
{daily_outcomes.to_dict(orient='records')}

Describe any notable trends over time: are drop-offs increasing or decreasing, 
are purchases spiking on certain days, etc.?
"""
    daily_narrative = generate_llm_response(daily_prompt)
    st.info(f"**Daily Trends Narrative**: {daily_narrative}")


# --------------------------------------------------------------------------- #
# 4) SCENARIO PLANNING
# --------------------------------------------------------------------------- #
st.subheader("Scenario Planning / 'What-If' Analysis")

if scenario_run and scenario_text.strip():
    scenario_prompt = f"""
Given our historical user journey data with multiple channels and devices, 
please forecast or theorize what might happen if:
'{scenario_text}'

Focus on potential changes in funnel drop-offs and conversions. Provide data-informed reasoning.
"""
    scenario_response = generate_llm_response(scenario_prompt)
    st.success(f"**Scenario Analysis**:\n\n{scenario_response}")
elif scenario_run and not scenario_text.strip():
    st.warning("Please enter a scenario question before running analysis.")

st.write("---")
st.caption("Built with a large multi-channel dataset, Plotly, and your custom OpenAI client.")
