# ==========================================================
# Recruitment Efficiency Insight Dashboard (Final Version)
# ==========================================================
# Author: NeuraLens
# Purpose: Dashboard for recruitment analytics & prediction
# ==========================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------
st.set_page_config(page_title="Recruitment Efficiency Dashboard", layout="wide")
st.title("Recruitment Efficiency Insight Dashboard")

st.markdown("""
This dashboard helps HR teams analyze and predict recruitment efficiency across
departments, sources, and job roles. It uses three key metrics:

- **Time to Hire**
- **Cost per Hire**
- **Offer Acceptance Rate**
""")

# ----------------------------------------------------------
# MODEL HANDLING
# ----------------------------------------------------------
MODEL_DIR = "."
MODEL_DIR = "retrain_outputs"

import os
import joblib

models = {}
try:
    models["time_to_hire_days"] = joblib.load(os.path.join(MODEL_DIR, "model_time_to_hire_days_FEv3.pkl"))
    models["cost_per_hire"] = joblib.load(os.path.join(MODEL_DIR, "model_cost_per_hire_FEv3.pkl"))
    models["offer_acceptance_rate"] = joblib.load(os.path.join(MODEL_DIR, "model_offer_acceptance_rate_FEv3.pkl"))
    print("✅ All models loaded successfully.")
except Exception as e:
    print("❌ Error loading model files:", e)

MODEL_FILES = {
    "time": "model_time_to_hire_days_FEv3.pkl",
    "cost": "model_cost_per_hire_FEv3.pkl",
    "offer": "model_offer_acceptance_rate_FEv3.pkl",
}


@st.cache_resource
def load_models():
    models = {}
    missing = []
    for key, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            try:
                models[key] = joblib.load(path)
            except Exception:
                models[key] = None
        else:
            models[key] = None
            missing.append(fname)
    return models, missing


models, missing_models = load_models()
models_available = any(m is not None for m in models.values())

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data():
    if os.path.exists("final_recruitment_data_FEv3.csv"):
        return pd.read_csv("final_recruitment_data_FEv3.csv")
    else:
        return None


base_df = load_data()

# ----------------------------------------------------------
# EFFICIENCY SCORE CALCULATION
# ----------------------------------------------------------
def compute_efficiency(df):
    df = df.copy()
    for metric in ["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"]:
        if metric not in df.columns:
            df[metric] = np.nan

    df["time_score"] = 1 - (df["time_to_hire_days"] - df["time_to_hire_days"].min()) / (
        df["time_to_hire_days"].max() - df["time_to_hire_days"].min()
    )
    df["cost_score"] = 1 - (df["cost_per_hire"] - df["cost_per_hire"].min()) / (
        df["cost_per_hire"].max() - df["cost_per_hire"].min()
    )
    df["accept_score"] = (df["offer_acceptance_rate"] - df["offer_acceptance_rate"].min()) / (
        df["offer_acceptance_rate"].max() - df["offer_acceptance_rate"].min()
    )
    df["efficiency_score"] = (
        0.4 * df["time_score"] + 0.3 * df["cost_score"] + 0.3 * df["accept_score"]
    )
    return df


# ----------------------------------------------------------
# CREATE TABS
# ----------------------------------------------------------
tab_exec, tab_dept, tab_source, tab_job, tab_top10, tab_predict = st.tabs([
    "Executive Summary",
    "Department Efficiency",
    "Source Efficiency",
    "Job Role Efficiency",
    "Top 10 Most Efficient Recruitments",
    "Batch Prediction"
])

# ----------------------------------------------------------
# EXECUTIVE SUMMARY
# ----------------------------------------------------------
with tab_exec:
    st.header("Recruitment KPI Scorecard — Executive Overview")

    if base_df is None:
        st.warning("Default dataset not found. Please upload via Batch Prediction tab.")
    else:
        df = compute_efficiency(base_df)

        avg_time = round(df["time_to_hire_days"].mean(), 1)
        avg_cost = round(df["cost_per_hire"].mean(), 1)
        avg_accept = round(df["offer_acceptance_rate"].mean() * 100, 1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Average Time to Hire (days)", f"{avg_time}")
        c2.metric("Average Cost per Hire ($)", f"{avg_cost:,}")
        c3.metric("Offer Acceptance Rate (%)", f"{avg_accept}%")

        st.divider()

        st.subheader("Most Efficient Highlights")
        dept_best = df.groupby("department")["efficiency_score"].mean().sort_values(ascending=False)
        src_best = df.groupby("source")["efficiency_score"].mean().sort_values(ascending=False)
        job_best = df.groupby("job_title")["efficiency_score"].mean().sort_values(ascending=False)

        col1, col2, col3 = st.columns(3)
        col1.metric("Most Efficient Department", dept_best.index[0])
        col2.metric("Most Efficient Job Title", job_best.index[0])
        col3.metric("Most Efficient Source", src_best.index[0])

# ----------------------------------------------------------
# DEPARTMENT EFFICIENCY
# ----------------------------------------------------------
with tab_dept:
    st.header("Department Efficiency Overview")
    if base_df is not None:
        df = compute_efficiency(base_df)
        dept_summary = (
            df.groupby("department")[["time_to_hire_days", "cost_per_hire",
                                      "offer_acceptance_rate", "efficiency_score"]]
            .mean()
            .sort_values("efficiency_score", ascending=False)
        )
        st.dataframe(dept_summary.reset_index(), use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# SOURCE EFFICIENCY
# ----------------------------------------------------------
with tab_source:
    st.header("Source Efficiency Overview")
    if base_df is not None:
        df = compute_efficiency(base_df)
        src_summary = (
            df.groupby("source")[["time_to_hire_days", "cost_per_hire",
                                  "offer_acceptance_rate", "efficiency_score"]]
            .mean()
            .sort_values("efficiency_score", ascending=False)
        )
        st.dataframe(src_summary.reset_index(), use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# JOB ROLE EFFICIENCY
# ----------------------------------------------------------
with tab_job:
    st.header("Job Role Efficiency Overview")
    if base_df is not None:
        df = compute_efficiency(base_df)
        job_summary = (
            df.groupby("job_title")[["time_to_hire_days", "cost_per_hire",
                                     "offer_acceptance_rate", "efficiency_score"]]
            .mean()
            .sort_values("efficiency_score", ascending=False)
        )
        st.dataframe(job_summary.reset_index(), use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# TOP 10 MOST EFFICIENT RECRUITMENTS
# ----------------------------------------------------------
with tab_top10:
    st.header("Top 10 Most Efficient Recruitments")
    if base_df is not None:
        df = compute_efficiency(base_df)
        top10 = df.sort_values("efficiency_score", ascending=False).head(10)
        st.dataframe(
            top10[["department", "source", "job_title",
                   "time_to_hire_days", "cost_per_hire",
                   "offer_acceptance_rate", "efficiency_score"]],
            use_container_width=True, hide_index=True,
        )

# ==========================================================
# TAB 6: BATCH PREDICTION — Simplified Display
# ==========================================================
with tab6:
    st.header("Batch Prediction — Predict Recruitment KPIs from CSV")

    uploaded_file = st.file_uploader("Upload your recruitment dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(data[["department", "job_title", "time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"]].head(),
                     use_container_width=True)

        if models:
            try:
                # Run predictions
                data["pred_time_to_hire_days"] = models["time"].predict(data)
                data["pred_cost_per_hire"] = models["cost"].predict(data)
                data["pred_offer_acceptance_rate"] = models["offer"].predict(data)

                st.success("✅ Prediction completed successfully.")

                # Select key columns for display
                display_cols = [
                    "department", "job_title",
                    "time_to_hire_days", "pred_time_to_hire_days",
                    "cost_per_hire", "pred_cost_per_hire",
                    "offer_acceptance_rate", "pred_offer_acceptance_rate"
                ]

                st.subheader("Prediction Results (Key Metrics)")
                st.dataframe(data[display_cols].head(10), use_container_width=True)

                # Summary Metrics
                st.subheader("Summary Statistics (Predictions)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Predicted Time to Hire (days)", f"{data['pred_time_to_hire_days'].mean():.1f}")
                col2.metric("Avg Predicted Cost per Hire ($)", f"{data['pred_cost_per_hire'].mean():,.0f}")
                col3.metric("Avg Predicted Offer Acceptance Rate (%)", f"{data['pred_offer_acceptance_rate'].mean() * 100:.1f}%")

                # Downloadable CSV
                csv = data[display_cols].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name="recruitment_predictions_simplified.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("Models not loaded. Please check your .pkl files in GitHub or Streamlit environment.")
    else:
        st.info("Upload a CSV file to run predictions.")
