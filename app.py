# ==========================================================
# Recruitment Efficiency Insight Dashboard
# ==========================================================
# Author: Patricia Prinza
# Purpose: Data-Driven HR Insight — Department, Source, Job Title, and Efficiency Ranking
# ==========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------
# Page setup
# ----------------------------------------------------------
st.set_page_config(
    page_title="Recruitment Efficiency Insight — FEv3",
    page_icon="💼",
    layout="wide"
)

st.title("Recruitment Efficiency Insight Dashboard")
st.markdown("""
This dashboard helps HR teams understand **which departments, job titles, and candidate sources**
contribute most to recruitment efficiency.

> Focus Metrics: **Time to Hire**, **Cost per Hire**, and **Offer Acceptance Rate**
""")

# ----------------------------------------------------------
# Load dataset
# ----------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("final_recruitment_data_FEv3.csv")
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception as e:
        st.error(f"Unable to load dataset: {e}")
        return None

df = load_data()
if df is None:
    st.stop()

# Sidebar data check
st.sidebar.header("Data Info")
st.sidebar.write("Detected Columns:")
st.sidebar.write(list(df.columns))

required_cols = [
    "department", "source", "job_title",
    "time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"
]
missing = [col for col in required_cols if col not in df.columns]

if missing:
    st.error(f"Missing columns in dataset: {missing}")
    st.info("Please ensure your CSV has the correct columns.")
    st.stop()

# ==========================================================
# Tabs for analysis
# ==========================================================
tabs = st.tabs([
    "Department Efficiency",
    "Source Effectiveness",
    "Job Title Complexity",
    "Overall Efficiency Ranking"
])

# ==========================================================
# TAB 1 — Department Analysis
# ==========================================================
with tabs[0]:
    st.subheader("Department Efficiency Overview")

    dept_summary = (
        df.groupby("department")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"]]
        .mean()
        .round(2)
        .sort_values("time_to_hire_days")
    )

    best_dept_time = dept_summary["time_to_hire_days"].idxmin()
    best_dept_cost = dept_summary["cost_per_hire"].idxmin()
    best_dept_accept = dept_summary["offer_acceptance_rate"].idxmax()

    st.markdown(f"""
    **Key Insights:**
    - Fastest Hiring Department: **{best_dept_time}**
    - Lowest Cost per Hire: **{best_dept_cost}**
    - Highest Offer Acceptance: **{best_dept_accept}**
    """)

    st.markdown("### Department Efficiency Metrics")
    st.dataframe(dept_summary, use_container_width=True)

    # Visualization
    st.markdown("### 📊 Average Time to Hire by Department")
    fig, ax = plt.subplots(figsize=(6,3))
    dept_summary["time_to_hire_days"].sort_values().plot(kind="bar", ax=ax)
    ax.set_ylabel("Days")
    ax.set_xlabel("Department")
    ax.set_title("Average Time to Hire by Department")
    st.pyplot(fig)

# ==========================================================
# TAB 2 — Source Analysis
# ==========================================================
with tabs[1]:
    st.subheader("🔗 Candidate Source Effectiveness")

    src_summary = (
        df.groupby("source")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"]]
        .mean()
        .round(2)
        .sort_values("cost_per_hire")
    )

    best_src_time = src_summary["time_to_hire_days"].idxmin()
    best_src_cost = src_summary["cost_per_hire"].idxmin()
    best_src_accept = src_summary["offer_acceptance_rate"].idxmax()

    st.markdown(f"""
    **Key Insights:**
    - Fastest Source: **{best_src_time}**
    - Most Cost-Efficient Source: **{best_src_cost}**
    - Highest Offer Acceptance Source: **{best_src_accept}**
    """)

    st.markdown("### Source Efficiency Metrics")
    st.dataframe(src_summary, use_container_width=True)

    # Visualization
    st.markdown("### 📊 Average Cost per Hire by Source")
    fig, ax = plt.subplots(figsize=(6,3))
    src_summary["cost_per_hire"].sort_values().plot(kind="bar", ax=ax, color="#2ECC71")
    ax.set_ylabel("Cost ($)")
    ax.set_xlabel("Source")
    ax.set_title("Average Cost per Hire by Source")
    st.pyplot(fig)

# ==========================================================
# TAB 3 — Job Title Analysis
# ==========================================================
with tabs[2]:
    st.subheader("Job Title Complexity & Efficiency")

    job_summary = (
        df.groupby("job_title")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"]]
        .mean()
        .round(2)
        .sort_values("time_to_hire_days")
    )

    hardest_job = job_summary["time_to_hire_days"].idxmax()
    most_costly_job = job_summary["cost_per_hire"].idxmax()
    best_accept_job = job_summary["offer_acceptance_rate"].idxmax()

    st.markdown(f"""
    **Key Insights:**
    - Longest Hiring Duration: **{hardest_job}**
    - Most Expensive Role to Hire: **{most_costly_job}**
    - Highest Offer Acceptance Role: **{best_accept_job}**
    """)

    st.markdown("### Job Title Efficiency Metrics")
    st.dataframe(job_summary, use_container_width=True)

    # Visualization
    st.markdown("### 📊 Offer Acceptance Rate by Job Title")
    fig, ax = plt.subplots(figsize=(6,3))
    job_summary["offer_acceptance_rate"].sort_values(ascending=False).plot(kind="bar", ax=ax, color="#3498DB")
    ax.set_ylabel("Acceptance Rate")
    ax.set_xlabel("Job Title")
    ax.set_title("Average Offer Acceptance Rate by Job Title")
    st.pyplot(fig)

# ==========================================================
# TAB 4 — Overall Efficiency Ranking
# ==========================================================
with tabs[3]:
    st.subheader("Overall Department Efficiency Ranking")

    # Normalize each metric (scale 0–1)
    norm_df = df.copy()
    summary = (
        norm_df.groupby("department")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"]]
        .mean()
        .reset_index()
    )

    # Normalization (0 = poor, 1 = excellent)
    summary["time_score"] = 1 - (summary["time_to_hire_days"] - summary["time_to_hire_days"].min()) / (
        summary["time_to_hire_days"].max() - summary["time_to_hire_days"].min()
    )
    summary["cost_score"] = 1 - (summary["cost_per_hire"] - summary["cost_per_hire"].min()) / (
        summary["cost_per_hire"].max() - summary["cost_per_hire"].min()
    )
    summary["accept_score"] = (summary["offer_acceptance_rate"] - summary["offer_acceptance_rate"].min()) / (
        summary["offer_acceptance_rate"].max() - summary["offer_acceptance_rate"].min()
    )

    # Combined efficiency score (weighted average)
    summary["efficiency_score"] = (
        0.4 * summary["time_score"] +
        0.3 * summary["cost_score"] +
        0.3 * summary["accept_score"]
    )

    summary = summary.sort_values("efficiency_score", ascending=False).reset_index(drop=True)

    st.markdown("""
    The **Efficiency Score** is a composite metric combining:
    - 40% Time Efficiency (faster hiring = better)
    - 30% Cost Efficiency (lower cost = better)
    - 30% Offer Efficiency (higher acceptance = better)
    """)

    st.dataframe(summary[["department", "time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
                 .round(2), use_container_width=True)

    # Visualization: Efficiency Ranking
    st.markdown("###Efficiency Ranking by Department")
    fig, ax = plt.subplots(figsize=(7,4))
    summary.plot(x="department", y="efficiency_score", kind="barh", ax=ax, color="#F1C40F")
    ax.set_xlabel("Efficiency Score")
    ax.set_ylabel("Department")
    ax.invert_yaxis()
    ax.set_title("Overall Recruitment Efficiency Score by Department")
    st.pyplot(fig)

# ==========================================================
# Footer
# ==========================================================
st.markdown("---")
st.caption("Data-Driven Recruitment Efficiency Dashboard (Department, Source, Job Title, and Overall Efficiency)")
