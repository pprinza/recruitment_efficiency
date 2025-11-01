# ==========================================================
# Recruitment Efficiency Insight Dashboard
# Created by: NeuraLens
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np

# --- Page setup ---
st.set_page_config(page_title="Recruitment Efficiency Insight", page_icon="💼", layout="wide")

st.title("Recruitment Efficiency Insight Dashboard")
st.markdown("""
This dashboard helps HR teams understand **which departments, job titles, and candidate sources**
contribute most to recruitment efficiency.

> Focus Metrics: **Time to Hire**, **Cost per Hire**, and **Offer Acceptance Rate**
""")

# --- Load dataset ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("deployment_retrain_summary_FEv3.csv")
        df.columns = df.columns.str.strip().str.lower()  # Normalize column names
        return df
    except Exception as e:
        st.error(f"Unable to load data: {e}")
        return None

df = load_data()
if df is None:
    st.stop()

# --- Column diagnostic ---
st.sidebar.subheader("Data Check")
st.sidebar.write("Detected columns:")
st.sidebar.write(list(df.columns))

required_cols = ["department", "source", "job_title", "time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"]
missing = [col for col in required_cols if col not in df.columns]

if missing:
    st.error(f"Missing columns in dataset: {missing}")
    st.info("Please ensure your CSV has the correct columns.")
    st.stop()

# --- Tabs ---
tabs = st.tabs(["🏢 Department Efficiency", "🔗 Source Effectiveness", "🧩 Job Title Complexity"])

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

# ==========================================================
# Footer
# ==========================================================
st.markdown("---")
st.caption("Data-Driven Recruitment Efficiency Dashboard (Department, Source, and Job Insights)")
