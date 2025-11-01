# ==========================================================
# Recruitment Efficiency Insight Dashboard (FEv3++)
# ==========================================================
# Author: NeuraLens
# Purpose: Data-Driven HR Insight — Department, Source, Job Title, and Individual Efficiency Ranking
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Page setup
# ----------------------------------------------------------
st.set_page_config(
    page_title="Recruitment Efficiency Insight — FEv3++",
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

# Sidebar info
st.sidebar.header("Data Info")
st.sidebar.write("Detected Columns:")
st.sidebar.write(list(df.columns))

required_cols = [
    "recruitment_id", "department", "source", "job_title",
    "time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"
]
missing = [col for col in required_cols if col not in df.columns]

if missing:
    st.error(f"Missing columns in dataset: {missing}")
    st.info("Please ensure your CSV has the correct columns.")
    st.stop()

# ==========================================================
# Tabs
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
    st.subheader("🏢 Department Efficiency Overview")

    dept_summary = (
        df.groupby("department")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"]]
        .mean()
        .round(2)
        .sort_values("time_to_hire_days")
        .reset_index()
    )

    best_dept_time = dept_summary.loc[dept_summary["time_to_hire_days"].idxmin(), "department"]
    best_dept_cost = dept_summary.loc[dept_summary["cost_per_hire"].idxmin(), "department"]
    best_dept_accept = dept_summary.loc[dept_summary["offer_acceptance_rate"].idxmax(), "department"]

    st.markdown(f"""
    **Highlights:**
    - Fastest hiring department: **{best_dept_time}**
    - Lowest cost per hire: **{best_dept_cost}**
    - Highest offer acceptance: **{best_dept_accept}**
    """)

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
        .reset_index()
    )

    best_src_time = src_summary.loc[src_summary["time_to_hire_days"].idxmin(), "source"]
    best_src_cost = src_summary.loc[src_summary["cost_per_hire"].idxmin(), "source"]
    best_src_accept = src_summary.loc[src_summary["offer_acceptance_rate"].idxmax(), "source"]

    st.markdown(f"""
    **Highlights:**
    - Fastest source: **{best_src_time}**
    - Most cost-efficient source: **{best_src_cost}**
    - Highest offer acceptance: **{best_src_accept}**
    """)

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
        .reset_index()
    )

    st.dataframe(job_summary, use_container_width=True)

# ==========================================================
# TAB 4 — Overall Efficiency Ranking (Enhanced)
# ==========================================================
with tabs[3]:
    st.subheader("Overall Recruitment Efficiency Ranking")

    # --- Department-level efficiency ---
    dept_summary = (
        df.groupby("department")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"]]
        .mean()
        .reset_index()
    )

    # Normalization (0–1 scale)
    dept_summary["time_score"] = 1 - (dept_summary["time_to_hire_days"] - dept_summary["time_to_hire_days"].min()) / (
        dept_summary["time_to_hire_days"].max() - dept_summary["time_to_hire_days"].min()
    )
    dept_summary["cost_score"] = 1 - (dept_summary["cost_per_hire"] - dept_summary["cost_per_hire"].min()) / (
        dept_summary["cost_per_hire"].max() - dept_summary["cost_per_hire"].min()
    )
    dept_summary["accept_score"] = (dept_summary["offer_acceptance_rate"] - dept_summary["offer_acceptance_rate"].min()) / (
        dept_summary["offer_acceptance_rate"].max() - dept_summary["offer_acceptance_rate"].min()
    )

    dept_summary["efficiency_score"] = (
        0.4 * dept_summary["time_score"] +
        0.3 * dept_summary["cost_score"] +
        0.3 * dept_summary["accept_score"]
    )

    dept_summary = dept_summary.sort_values("efficiency_score", ascending=False).reset_index(drop=True)

    st.markdown("### 🏢 Department Efficiency Ranking")
    st.dataframe(dept_summary.round(2), use_container_width=True)

    # --- Individual-level Top 10 Recruitments ---
    st.markdown("---")
    st.subheader("Top 10 Most Efficient Recruitments (Individual Level)")

    df_ind = df.copy()

    # Normalize each metric
    df_ind["time_score"] = 1 - (df_ind["time_to_hire_days"] - df_ind["time_to_hire_days"].min()) / (
        df_ind["time_to_hire_days"].max() - df_ind["time_to_hire_days"].min()
    )
    df_ind["cost_score"] = 1 - (df_ind["cost_per_hire"] - df_ind["cost_per_hire"].min()) / (
        df_ind["cost_per_hire"].max() - df_ind["cost_per_hire"].min()
    )
    df_ind["accept_score"] = (df_ind["offer_acceptance_rate"] - df_ind["offer_acceptance_rate"].min()) / (
        df_ind["offer_acceptance_rate"].max() - df_ind["offer_acceptance_rate"].min()
    )

    # Weighted efficiency score
    df_ind["efficiency_score"] = (
        0.4 * df_ind["time_score"] +
        0.3 * df_ind["cost_score"] +
        0.3 * df_ind["accept_score"]
    )

    top10 = (
        df_ind.sort_values("efficiency_score", ascending=False)
        [["recruitment_id", "department", "source", "job_title",
          "time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
        .head(10)
        .reset_index(drop=True)
        .round(2)
    )

    # Add ranking column
    top10.index = np.arange(1, len(top10) + 1)
    top10.index.name = "Rank"

    # Highlight Top 3
    def highlight_top3(row):
        if row.name <= 3:
            return ['background-color: #d4edda'] * len(row)
        return [''] * len(row)

    st.dataframe(
        top10.style.apply(highlight_top3, axis=1),
        use_container_width=True
    )

    # --- Visualization: efficiency distribution ---
    st.markdown("### 📈 Efficiency Score Distribution")
    fig, ax = plt.subplots(figsize=(6,3))
    df_ind["efficiency_score"].plot(kind="hist", bins=20, color="#F1C40F", ax=ax)
    ax.set_xlabel("Efficiency Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Recruitment Efficiency Scores")
    st.pyplot(fig)

# ==========================================================
# Footer
# ==========================================================
st.markdown("---")
st.caption("Data-Driven Recruitment Efficiency Dashboard (with Ranking & Top 10 Highlight)")
