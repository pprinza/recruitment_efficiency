# ==========================================================
# Recruitment Efficiency Insight Dashboard
# ==========================================================
# Author: NeuraLens
# Purpose: Data-Driven HR Insight — Department, Source, Job Title, and Individual Efficiency Ranking
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np

# ----------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------
st.set_page_config(page_title="Recruitment Efficiency Dashboard", layout="wide")
st.title("Recruitment Efficiency Insight Dashboard")

st.markdown("""
This dashboard helps HR teams understand which departments, sources, and job roles 
contribute most to overall recruitment efficiency based on three key metrics:

- Time to Hire  
- Cost per Hire  
- Offer Acceptance Rate
""")

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_recruitment_data_FEv3.csv")
    return df

df = load_data()

required_cols = ['department', 'source', 'job_title', 'time_to_hire_days', 'cost_per_hire', 'offer_acceptance_rate']
if not all(col in df.columns for col in required_cols):
    st.error(f"Missing columns in dataset: {set(required_cols) - set(df.columns)}")
    st.stop()

# ----------------------------------------------------------
# CALCULATE EFFICIENCY SCORE
# ----------------------------------------------------------
def compute_efficiency(df):
    df = df.copy()
    df['time_score'] = 1 - (df['time_to_hire_days'] - df['time_to_hire_days'].min()) / (df['time_to_hire_days'].max() - df['time_to_hire_days'].min())
    df['cost_score'] = 1 - (df['cost_per_hire'] - df['cost_per_hire'].min()) / (df['cost_per_hire'].max() - df['cost_per_hire'].min())
    df['accept_score'] = (df['offer_acceptance_rate'] - df['offer_acceptance_rate'].min()) / (df['offer_acceptance_rate'].max() - df['offer_acceptance_rate'].min())
    df['efficiency_score'] = 0.4 * df['time_score'] + 0.3 * df['cost_score'] + 0.3 * df['accept_score']
    return df

df = compute_efficiency(df)

# ----------------------------------------------------------
# CREATE TABS
# ----------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Summary",
    "Department Efficiency",
    "Source Efficiency",
    "Job Role Efficiency",
    "Top 10 Most Efficient Recruitments"
])

# ==========================================================
# EXECUTIVE SUMMARY
# ==========================================================
with tab1:
    st.header("Recruitment KPI Scorecard — Executive Overview")

    avg_time = round(df['time_to_hire_days'].mean())
    avg_cost = round(df['cost_per_hire'].mean())
    avg_accept = round(df['offer_acceptance_rate'].mean() * 100)

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Time to Hire (days)", f"{avg_time}")
    col2.metric("Average Cost per Hire ($)", f"{avg_cost:,}")
    col3.metric("Offer Acceptance Rate (%)", f"{avg_accept}%")

    st.divider()

    st.subheader("The Most Efficient Department")
    best_dept = (
        df.groupby("department")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
        .mean()
        .sort_values("efficiency_score", ascending=False)
    )
    st.dataframe(best_dept.head(1).reset_index(), use_container_width=True, hide_index=True)

    st.subheader("The Most Efficient Source")
    best_source = (
        df.groupby("source")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
        .mean()
        .sort_values("efficiency_score", ascending=False)
    )
    st.dataframe(best_source.head(1).reset_index(), use_container_width=True, hide_index=True)

    st.subheader("The Most Efficient Job Title")
    best_role = (
        df.groupby("job_title")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
        .mean()
        .sort_values("efficiency_score", ascending=False)
    )
    st.dataframe(best_role.head(1).reset_index(), use_container_width=True, hide_index=True)

# ==========================================================
# DEPARTMENT EFFICIENCY
# ==========================================================
with tab2:
    st.header("Department Efficiency Overview (Sorted by Efficiency Score)")
    dept_summary = (
        df.groupby("department")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
        .mean()
        .sort_values("efficiency_score", ascending=False)
    )
    st.dataframe(dept_summary.reset_index(), use_container_width=True, hide_index=True)

# ==========================================================
# SOURCE EFFICIENCY
# ==========================================================
with tab3:
    st.header("Source Efficiency Overview (Sorted by Efficiency Score)")
    source_summary = (
        df.groupby("source")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
        .mean()
        .sort_values("efficiency_score", ascending=False)
    )
    st.dataframe(source_summary.reset_index(), use_container_width=True, hide_index=True)

# ==========================================================
# JOB ROLE EFFICIENCY
# ==========================================================
with tab4:
    st.header("Job Role Efficiency Overview (Sorted by Efficiency Score)")
    job_summary = (
        df.groupby("job_title")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
        .mean()
        .sort_values("efficiency_score", ascending=False)
    )
    st.dataframe(job_summary.reset_index(), use_container_width=True, hide_index=True)

# ==========================================================
# TOP 10 MOST EFFICIENT RECRUITMENTS
# ==========================================================
with tab5:
    st.header("Top 10 Most Efficient Recruitments (Individual Level)")
    top10 = df.sort_values("efficiency_score", ascending=False).head(10)
    top10_display = top10[
        ["department", "source", "job_title", "time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]
    ]
    st.dataframe(top10_display.reset_index(drop=True), use_container_width=True, hide_index=True)
