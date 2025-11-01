# ==========================================================
# Recruitment Efficiency Insight Dashboard
# ==========================================================
# Author: NeuraLens
# Purpose: Data-Driven HR Insight — Department, Source, Job Title, and Individual Efficiency Ranking
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Recruitment Efficiency Dashboard", page_icon="💼", layout="wide")

st.title("Recruitment Efficiency Insight Dashboard")
st.markdown("""
This dashboard helps HR teams understand **which departments, job titles, and candidate sources** contribute most to recruitment efficiency.

**Focus Metrics:** Time to Hire, Cost per Hire, and Offer Acceptance Rate
""")

# -------------------------------
# Load data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_recruitment_data_FEv3.csv")
    return df

df = load_data()

required_cols = ['department', 'source', 'job_title', 'time_to_hire_days', 'cost_per_hire', 'offer_acceptance_rate']
if not all(col in df.columns for col in required_cols):
    st.error(f"Missing columns in dataset: {set(required_cols) - set(df.columns)}")
    st.info("Please ensure your CSV has the correct columns.")
    st.stop()

# -------------------------------
# Calculate Efficiency Score
# -------------------------------
def compute_efficiency_score(df):
    df = df.copy()
    df['time_score'] = 1 - (df['time_to_hire_days'] - df['time_to_hire_days'].min()) / (df['time_to_hire_days'].max() - df['time_to_hire_days'].min())
    df['cost_score'] = 1 - (df['cost_per_hire'] - df['cost_per_hire'].min()) / (df['cost_per_hire'].max() - df['cost_per_hire'].min())
    df['accept_score'] = (df['offer_acceptance_rate'] - df['offer_acceptance_rate'].min()) / (df['offer_acceptance_rate'].max() - df['offer_acceptance_rate'].min())
    df['efficiency_score'] = 0.4 * df['time_score'] + 0.3 * df['cost_score'] + 0.3 * df['accept_score']
    return df

df = compute_efficiency_score(df)

# -------------------------------
# Tabs setup
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Summary Dashboard",
    "Department Efficiency",
    "Source Effectiveness",
    "Job Title Complexity",
    "Overall Efficiency Ranking"
])

# ==========================================================
# TAB 1 — Executive Summary Dashboard
# ==========================================================
with tab1:
    st.header("Recruitment KPI Scorecard — Executive Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Time to Hire (days)", f"{df['time_to_hire_days'].mean():.1f}")
    col2.metric("Average Cost per Hire ($)", f"{df['cost_per_hire'].mean():,.0f}")
    col3.metric("Offer Acceptance Rate (%)", f"{df['offer_acceptance_rate'].mean() * 100:.1f}%")

    st.markdown("###Top 5 Most Efficient Departments")
    dept_eff = df.groupby("department")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]].mean().sort_values("efficiency_score", ascending=False).head(5)
    st.dataframe(dept_eff.style.background_gradient(cmap="Greens"), use_container_width=True)

    st.markdown("###Top 5 Most Cost-Effective Sources")
    src_eff = df.groupby("source")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]].mean().sort_values("cost_per_hire").head(5)
    st.dataframe(src_eff.style.background_gradient(cmap="Blues"), use_container_width=True)

# ==========================================================
# TAB 2 — Department Efficiency
# ==========================================================
with tab2:
    st.header("Department Efficiency Overview")
    dept_summary = df.groupby("department")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]].mean()
    dept_summary = dept_summary.sort_values("efficiency_score", ascending=False)
    st.dataframe(dept_summary.style.background_gradient(cmap="Greens"), use_container_width=True)

# ==========================================================
# TAB 3 — Source Effectiveness
# ==========================================================
with tab3:
    st.header("Source Effectiveness & Cost Efficiency")
    source_summary = df.groupby("source")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]].mean()
    source_summary = source_summary.sort_values("efficiency_score", ascending=False)
    st.dataframe(source_summary.style.background_gradient(cmap="Blues"), use_container_width=True)

# ==========================================================
# TAB 4 — Job Title Complexity
# ==========================================================
with tab4:
    st.header("Job Title Complexity & Efficiency")
    job_summary = df.groupby("job_title")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]].mean()
    job_summary = job_summary.sort_values("efficiency_score", ascending=False)
    st.dataframe(job_summary.style.background_gradient(cmap="Oranges"), use_container_width=True)

# ==========================================================
# TAB 5 — Overall Efficiency Ranking
# ==========================================================
with tab5:
    st.header("Overall Efficiency Ranking — Top 10 Most Efficient Recruitments")

    top10 = df.sort_values("efficiency_score", ascending=False).head(10)
    top10_display = top10[[
        "department", "source", "job_title",
        "time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"
    ]]

    st.dataframe(top10_display.style.applymap(
        lambda v: 'background-color: #d4edda' if v == top10_display['efficiency_score'].max() else ''
    ), use_container_width=True)

    st.caption("Highest efficiency scores indicate the most optimized balance between speed, cost, and offer acceptance rate.")
