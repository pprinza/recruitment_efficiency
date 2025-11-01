# ==========================================================
# Recruitment Efficiency Insight Dashboard
# ==========================================================
# Author: NeuraLens
# Purpose: Dnified Streamlit Dashboard for Analytical & Predictive Insight
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------
st.set_page_config(page_title="Recruitment Efficiency Dashboard", layout="wide")
st.title("Recruitment Efficiency Insight & Prediction App")

st.markdown("""
This application combines **analytical insights** and **real-time prediction**  
based on the FEv3 Machine Learning model.

Key objectives:
- Reduce Time to Hire  
- Optimize Cost Allocation  
- Improve Candidate Engagement and Offer Acceptance  
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
# COMPUTE EFFICIENCY SCORE
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
# LOAD TRAINED MODELS (for Prediction Tab)
# ----------------------------------------------------------
@st.cache_resource
def load_models():
    MODEL_DIR = "retrain_outputs"
    models = {
        "time": joblib.load(os.path.join(MODEL_DIR, "model_time_to_hire_days_FEv3.pkl")),
        "cost": joblib.load(os.path.join(MODEL_DIR, "model_cost_per_hire_FEv3.pkl")),
        "offer": joblib.load(os.path.join(MODEL_DIR, "model_offer_acceptance_rate_FEv3.pkl"))
    }
    return models

models = load_models()

# ----------------------------------------------------------
# CREATE TABS
# ----------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Executive Summary",
    "Department Efficiency",
    "Source Efficiency",
    "Job Role Efficiency",
    "Top 10 Most Efficient Recruitments",
    "Prediction Tool"
])

# ==========================================================
# TAB 1 — EXECUTIVE SUMMARY
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
    st.subheader("Top Performers")

    best_dept = (
        df.groupby("department")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
        .mean().sort_values("efficiency_score", ascending=False)
    ).head(1)

    best_source = (
        df.groupby("source")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
        .mean().sort_values("efficiency_score", ascending=False)
    ).head(1)

    best_role = (
        df.groupby("job_title")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
        .mean().sort_values("efficiency_score", ascending=False)
    ).head(1)

    st.write("**Most Efficient Department:**")
    st.dataframe(best_dept.reset_index(), hide_index=True, use_container_width=True)

    st.write("**Most Efficient Source:**")
    st.dataframe(best_source.reset_index(), hide_index=True, use_container_width=True)

    st.write("**Most Efficient Job Role:**")
    st.dataframe(best_role.reset_index(), hide_index=True, use_container_width=True)

# ==========================================================
# TAB 2 — DEPARTMENT EFFICIENCY
# ==========================================================
with tab2:
    st.header("Department Efficiency (sorted by efficiency score)")
    dept_summary = (
        df.groupby("department")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
        .mean().sort_values("efficiency_score", ascending=False)
    )
    st.dataframe(dept_summary.reset_index(), use_container_width=True, hide_index=True)

# ==========================================================
# TAB 3 — SOURCE EFFICIENCY
# ==========================================================
with tab3:
    st.header("Source Efficiency (sorted by efficiency score)")
    source_summary = (
        df.groupby("source")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
        .mean().sort_values("efficiency_score", ascending=False)
    )
    st.dataframe(source_summary.reset_index(), use_container_width=True, hide_index=True)

# ==========================================================
# TAB 4 — JOB ROLE EFFICIENCY
# ==========================================================
with tab4:
    st.header("Job Role Efficiency (sorted by efficiency score)")
    job_summary = (
        df.groupby("job_title")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
        .mean().sort_values("efficiency_score", ascending=False)
    )
    st.dataframe(job_summary.reset_index(), use_container_width=True, hide_index=True)

# ==========================================================
# TAB 5 — TOP 10 MOST EFFICIENT RECRUITMENTS
# ==========================================================
with tab5:
    st.header("Top 10 Most Efficient Recruitments (Individual Level)")
    top10 = df.sort_values("efficiency_score", ascending=False).head(10)
    top10_display = top10[
        ["department", "source", "job_title", "time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]
    ]
    st.dataframe(top10_display.reset_index(drop=True), use_container_width=True, hide_index=True)

# ==========================================================
# TAB 6 — PREDICTION TOOL
# ==========================================================
with tab6:
    st.header("Predict Recruitment Outcomes")

    with st.form("prediction_form"):
        department = st.selectbox("Department", ["Engineering", "Product", "HR", "Sales", "Marketing", "Finance"])
        source = st.selectbox("Source", ["Referral", "LinkedIn", "Recruiter", "Job Portal"])
        process_efficiency = st.slider("Process Efficiency", 0.0, 1.0, 0.7)
        cost_intensity = st.slider("Cost Intensity", 0.0, 1.0, 0.5)
        engagement_score = st.slider("Engagement Score", 0.0, 1.0, 0.6)
        dept_efficiency = st.slider("Department Efficiency", 0.0, 1.0, 0.8)
        offer_readiness = st.slider("Offer Readiness", 0.0, 1.0, 0.75)
        candidate_satisfaction = st.slider("Candidate Satisfaction", 0.0, 1.0, 0.7)
        submitted = st.form_submit_button("Run Prediction")

    if submitted:
        input_data = pd.DataFrame([{
            "department": department,
            "source": source,
            "process_efficiency": process_efficiency,
            "cost_intensity": cost_intensity,
            "engagement_score": engagement_score,
            "dept_efficiency": dept_efficiency,
            "offer_readiness": offer_readiness,
            "candidate_satisfaction": candidate_satisfaction
        }])

        time_pred = models["time"].predict(input_data)[0]
        cost_pred = models["cost"].predict(input_data)[0]
        offer_pred = models["offer"].predict(input_data)[0]

        # Avoid negatives or >100%
        time_pred = max(0, time_pred)
        cost_pred = max(0, cost_pred)
        offer_pred = max(0, min(1, offer_pred))

        st.subheader("Predicted Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Time to Hire (days)", f"{time_pred:.1f}")
        col2.metric("Predicted Cost per Hire ($)", f"{cost_pred:,.2f}")
        col3.metric("Offer Acceptance Rate (%)", f"{offer_pred*100:.1f}")

        st.info("Prediction based on FEv3 model (Ridge, Lasso, XGBoost). Features standardized and preprocessed automatically.")
